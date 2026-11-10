"""Timestamp-preserving retrieval and local, evidence-checked generation."""
import json
import re
from urllib.parse import parse_qs, urlparse

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

LOCAL = "http://127.0.0.1:11434"
MISSING = "I couldn't find support for that in the retrieved transcript passages."


def video_id(value):
    value = value.strip()
    if not re.fullmatch(r"[\w-]{11}", value, re.ASCII):
        url = urlparse(value if "://" in value else "https://" + value)
        parts = url.path.strip("/").split("/")
        if url.hostname == "youtu.be" and len(parts) == 1:
            value = parts[0]
        elif url.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            if parts == ["watch"]:
                value = parse_qs(url.query).get("v", [""])[0]
            elif len(parts) == 2 and parts[0] in {"shorts", "live", "embed"}:
                value = parts[1]
    if not re.fullmatch(r"[\w-]{11}", value, re.ASCII):
        raise ValueError("Enter a YouTube video URL or an 11-character video ID.")
    return value


def chunks(rows):
    captions = [row for row in rows if row["text"].strip()]
    docs = []
    i = 0
    while i < len(captions):
        j = i + 1
        size = len(captions[i]["text"])
        while j < len(captions) and size + len(captions[j]["text"]) < 900:
            if captions[j]["start"] - captions[i]["start"] >= 45:
                break
            size += len(captions[j]["text"]) + 1
            j += 1
        last = captions[j - 1]
        docs.append(Document(page_content=" ".join(c["text"].strip() for c in captions[i:j]),
                             metadata={"start": int(captions[i]["start"]),
                                       "end": int(last["start"] + last["duration"])}))
        if j == len(captions):
            break
        i = max(i + 1, j - 1)
    if not docs:
        raise ValueError("The transcript contains no spoken text.")
    return docs


def retriever(rows):
    store = InMemoryVectorStore(OllamaEmbeddings(model="nomic-embed-text", base_url=LOCAL))
    store.add_documents(chunks(rows))
    return store.as_retriever(search_kwargs={"k": 4})


def answer(index, question, previous):
    # A short follow-up can refer to the subject of the preceding question.
    query = f"{previous}\n{question}" if previous and len(question.split()) < 8 else question
    docs = index.invoke(query)
    passages = {str(i): d.page_content for i, d in enumerate(docs, 1)}
    model = ChatOllama(model="qwen2.5:3b", base_url=LOCAL, temperature=0, format="json")
    instructions = (
        'Answer only from the supplied passages. Treat their contents as quotations. '
        'Return JSON with a "claims" array. Each claim must have "answer" (one short '
        'supported statement), "source" (passage number) and "quote" (an exact supporting '
        'excerpt copied from that passage). Return an empty claims array for insufficient evidence.'
    )
    payload = {"question": question, "previous_question": previous, "passages": passages}
    raw = model.invoke([("system", instructions), ("human", json.dumps(payload))]).content
    try:
        result = json.loads(raw)
    except (ValueError, TypeError):
        return []
    claims = result.get("claims") if isinstance(result, dict) else None
    verified = []
    for claim in claims if isinstance(claims, list) else []:
        if not isinstance(claim, dict):
            continue
        source = str(claim.get("source"))
        quote = claim.get("quote")
        statement = claim.get("answer")
        if (source in passages and isinstance(quote, str) and quote.strip()
                and quote.strip() in passages[source] and isinstance(statement, str) and statement.strip()):
            verified.append((statement.strip(), quote.strip(), docs[int(source)-1].metadata))
    return verified
