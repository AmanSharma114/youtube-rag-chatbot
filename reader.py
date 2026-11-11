"""Optional local SQuAD2 reader. Run python reader.py once to install its weights."""
import os
import hashlib
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import requests

MODEL_DIR = Path(os.getenv("CUEFRAME_READER_DIR", Path.home() / ".cueframe" / "reader"))
BASE = "https://huggingface.co/optimum/roberta-base-squad2/resolve/db07770d7fa024df076e60947b445b0b3290f8eb"
HASHES = {
    "model.onnx": "204a84036a76ec7dafad3a4613f9425316b0308d77ad88326795fef6eb64b718",
    "tokenizer.json": "fb35154b5c214e4235de078bfd0890777e06bce973689112c1d06a9157ed0288",
}


@lru_cache(maxsize=1)
def load_reader():
    if not all((MODEL_DIR / name).is_file() for name in ("tokenizer.json", "model.onnx")):
        return None
    import onnxruntime as ort
    from tokenizers import Tokenizer

    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    session = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), options,
                                  providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512, strategy="only_second")
    return session, tokenizer


def read_spans(question, docs):
    reader = load_reader()
    if reader is None:
        return []
    session, tokenizer = reader
    # Bound the query independently to leave room for transcript tokens.
    question = tokenizer.decode(tokenizer.encode(question, add_special_tokens=False).ids[:100])
    found = []
    for doc in docs:
        encoding = tokenizer.encode(question, doc.page_content)
        ids = np.asarray([encoding.ids], dtype=np.int64)
        arrays = {"input_ids":ids, "attention_mask":np.ones_like(ids),
                  "token_type_ids":np.asarray([encoding.type_ids], dtype=np.int64)}
        starts, ends = session.run(None, {i.name:arrays[i.name] for i in session.get_inputs()})
        positions = np.array([i for i, side in enumerate(encoding.sequence_ids) if side == 1])
        if not len(positions):
            continue
        scores = starts[0, positions, None] + ends[0, positions][None, :]
        distance = positions[None, :] - positions[:, None]
        scores[(distance < 0) | (distance > 65)] = -np.inf
        a, b = np.unravel_index(np.argmax(scores), scores.shape)
        margin = float(scores[a,b] - starts[0,0] - ends[0,0])
        begin = encoding.offsets[positions[a]][0]
        end = encoding.offsets[positions[b]][1]
        text = doc.page_content[begin:end].strip()
        if text and margin > 0:
            found.append((margin, text, doc, begin, end))
    return sorted(found, key=lambda item:item[0], reverse=True)


def install():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        target = MODEL_DIR / name
        if target.is_file():
            continue
        with NamedTemporaryFile(dir=MODEL_DIR, delete=False) as file:
            temp = Path(file.name)
        try:
            digest = hashlib.sha256()
            with requests.get(f"{BASE}/{name}?download=true", stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with temp.open("wb") as file:
                    for chunk in response.iter_content(1024*1024):
                        digest.update(chunk)
                        file.write(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Checksum mismatch downloading {name}.")
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
    load_reader.cache_clear()
    load_reader()
    print("Local transcript reader ready.")


if __name__ == "__main__":
    install()
