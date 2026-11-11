"""Timestamp-preserving retrieval and local, evidence-checked generation."""
import json
import math
import re
from collections import Counter
from urllib.parse import parse_qs, urlparse

from reader import read_spans

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

LOCAL = "http://127.0.0.1:11434"


class CitationError(RuntimeError):
    """The model returned an answer that could not be traced to the captions."""


STOP = set(("a an and are as at be by can did do does for from how i in is it me of on or say says "
            "that the they this to video was what when where which who why with you").split())


def words(text):
    return set(re.findall(r"[^\W_]+", text.casefold())) - STOP


def caption_passages(rows):
    passages = []
    parts = []
    for row in rows:
        text = row["text"].strip()
        if not text or re.fullmatch(r"\[.*?\]|♪+", text):
            continue
        if parts and (row["start"] - last_end > 4 or len(" ".join(parts)) > 420):
            passages.append((" ".join(parts), start, last_end))
            parts = []
        if not parts:
            start = row["start"]
        parts.append(text)
        last_end = row["start"] + row["duration"]
        if re.search(r"""[.!?]["'”’)]*$""", text):
            passages.append((" ".join(parts), start, last_end))
            parts = []
    if parts:
        passages.append((" ".join(parts), start, last_end))
    return passages


def overview_question(question):
    normalized = " ".join(re.findall(r"[a-z0-9]+", question.lower().replace("’", "'")))
    normalized = re.sub(r"^(?:please |can you |could you )+", "", normalized)
    normalized = re.sub(r"^(?:tell me |give me |explain )", "", normalized)
    normalized = re.sub(r"^(?:what(?: s| is| are)? |list )", "", normalized)
    normalized = re.sub(r"\b(?:the|this|that|a|an|of|in|entire|whole)\b", " ", normalized)
    normalized = " ".join(normalized.split())
    return bool(re.fullmatch(
        r"(?:(?:main |overall |key )?(?:topics?|subjects?|points?|takeaways?|ideas?|theme|gist|overview|summary)"
        r"(?: (?:video|talk|lecture))?|(?:video|talk|lecture)(?: (?:main topic|topic|about|on|cover|covers|summary))"
        r"|summari[sz]e(?: (?:video|talk|lecture|it))?|(?:video|talk|lecture) (?:mainly |mostly )?about"
        r"|does (?:video|talk|lecture) (?:cover|discuss))", normalized))


def overview(rows, title, question):
    """Anchor the topic in the introduction and check the title against the captions."""
    passages = []
    for doc in chunks(rows):
        for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", doc.page_content):
            text = match.group().strip()
            if len(words(text)) < 3:
                continue
            begin = match.start() + len(match.group()) - len(match.group().lstrip())
            spans = [span for span in doc.metadata["offsets"] if span[0] < match.end() and span[1] > begin]
            if spans:
                passages.append((text, spans[0][2], spans[-1][3]))
    if not passages:
        passages = caption_passages(rows)
    extra_stop = {"so", "just", "we", "our", "my", "very", "like", "yeah", "okay", "right",
                  "know", "topic", "about", "going", "really", "here", "there", "have", "your"}
    terms = [words(text) - extra_stop for text, _, _ in passages]
    frequency = Counter(t for group in terms for t in group)
    title_words = words(title) - extra_stop
    anchors = title_words & frequency.keys()
    # Common words such as "every" or "actually" alone cannot corroborate a title.
    supported_title = bool(title_words and len(anchors) >= min(2, len(title_words))
                           and not title.startswith("YouTube video "))
    duration = max(1, passages[-1][2] - passages[0][1])
    ranked = []
    for i, (text, start, _) in enumerate(passages):
        position = (start - passages[0][1]) / duration
        topic_cue = re.search(r"\b(topic of (?:this|the) video|(?:in this (?:video|talk|lesson)|today) "
                              r"(?:we|i)|(?:we.ll|we will) (?:cover|explore|discuss))\b", text.lower())
        recurring = sum(math.log1p(frequency[t]) for t in terms[i]) / max(1, len(terms[i])) ** .65
        score = recurring + 3 * len(terms[i] & anchors) + 3 * max(0, 1 - position * 4)
        if topic_cue and position < .3:
            score += 15
        ranked.append((score, i))
    ranked.sort(reverse=True)
    selected = [ranked[0][1]]
    detailed = re.search(r"summari[sz]e|summary|takeaways?|points?|topics|overview", question, re.I)
    for _, i in ranked[1:] if detailed else []:
        if len(selected) == 3:
            break
        if len(terms[i]) < 4 or any(abs(i-j) < 2 or
                len(terms[i] & terms[j]) / max(1, len(terms[i] | terms[j])) > .45 for j in selected):
            continue
        selected.append(i)
    claims = []
    for i in selected:
        text, start, end = passages[i]
        statement = f"The main topic is “{title.rstrip('.')}”." if not claims and supported_title else text
        claims.append((statement, text, {"start": int(start), "end": math.ceil(end)}))
    return claims


def excerpt_answer(rows, question, title=""):
    """Use whole-transcript evidence for overviews and focused retrieval for details."""
    passages = caption_passages(rows)
    if not passages:
        return []
    if overview_question(question):
        return overview(rows, title, question)
    terms = words(question)
    captions = [words(passage[0]) for passage in passages]
    frequency = Counter(term for caption in captions for term in caption)
    best_score, best = 0, None
    for passage, caption in zip(passages, captions):
        score = sum(math.log1p(len(passages) / frequency[t]) for t in terms & caption)
        score /= 1 + len(caption) / 100
        if score > best_score:
            best_score, best = score, passage
    if best is None:
        return []
    text, start, end = best
    return [(text, text, {"start": int(start), "end": math.ceil(end)})]


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
        offsets = []
        position = 0
        texts = []
        for caption in captions[i:j]:
            text = caption["text"].strip()
            texts.append(text)
            offsets.append((position, position + len(text), int(caption["start"]),
                            math.ceil(caption["start"] + caption["duration"])))
            position += len(text) + 1
        docs.append(Document(page_content=" ".join(texts), metadata={"offsets": offsets}))
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


def informative_claims(question, claims):
    # The model sometimes echoes the category in the question ("two tracks")
    # even though the quoted evidence names both items.
    return [(quote if words(statement) <= words(question) else statement, quote, timing)
            for statement, quote, timing in claims]


def chat_context(turns):
    """Limit history sent to the model to turns from the current video."""
    return [{"question": question, "answer": [claim[0][:350] for claim in informative_claims(question, claims[:4])]}
            for question, claims in turns[-3:]]


def evidence(index, question, history, title, timeline, model):
    query = question
    if history:
        prompt = ("Rewrite the last question for transcript search. Resolve references from the "
                  "conversation. Keep named entities and the user's intent. Return JSON with a query string.")
        response = model.invoke([("system", prompt), ("human", json.dumps(
            {"video": title, "conversation": history, "question": question}))])
        try:
            planned = json.loads(response.content).get("query")
            if isinstance(planned, str) and planned.strip():
                query = planned.strip()[:600]
        except (ValueError, AttributeError):
            pass

    if not timeline:
        return query, index.invoke(query)
    # Reciprocal rank fusion balances semantic matches with distinctive names and terminology.
    tokens = words(query)
    term_sets = [words(doc.page_content) for doc in timeline]
    counts = Counter(term for terms in term_sets for term in terms)
    lexical = [sum(math.log1p(len(timeline) / counts[t]) for t in terms & tokens) for terms in term_sets]
    scores = Counter()
    for rank, i in enumerate(sorted(range(len(timeline)), key=lexical.__getitem__, reverse=True)):
        if lexical[i]:
            scores[timeline[i].page_content] += 1 / (5 + rank)
    for rank, doc in enumerate(index.invoke(query)):
        scores[doc.page_content] += 1 / (5 + rank)
    chosen = sorted(timeline, key=lambda d: scores[d.page_content], reverse=True)[:5]
    if overview_question(question):
        chosen.extend(timeline[round(i * (len(timeline)-1) / 3)] for i in range(4))
    return query, list({doc.page_content:doc for doc in chosen}.values())


def answer(index, question, previous, *, title="", rows=None):
    history = previous or []
    if isinstance(history, str):
        history = [{"question": history, "answer": []}]
    model = ChatOllama(model="qwen2.5:3b", base_url=LOCAL, temperature=0, format="json", num_ctx=8192,
                       num_predict=650)
    query, docs = evidence(index, question, history, title, chunks(rows) if rows else [], model)
    spans = read_spans(query, docs)
    multi = re.search(r"\b(compare|between|list|summari[sz]e|summary|both|two|three|four|five|all|tracks|differences|steps|tasks)\b",
                      question, re.I)
    # For a factual question a high-confidence span is a grounded answer on its own.
    if spans and spans[0][0] >= 4 and not multi and not overview_question(question):
        _, text, doc, begin, end = spans[0]
        offsets = [s for s in doc.metadata["offsets"] if s[0] < end and s[1] > begin]
        quote = doc.page_content[offsets[0][0]:offsets[-1][1]]
        return [(text, quote, {"start":offsets[0][2], "end":offsets[-1][3]})]

    # Preserve whole neighboring captions when a list or explanation crosses a boundary.
    selected = [item[2] for item in spans] + docs
    selected = list({doc.page_content:doc for doc in selected}.values())[:6]
    prompt = ('Answer the question directly using the transcript. Use conversation only to resolve '
              'references. For a list or comparison, name the distinct supported items. Use the reader '
              'hints only when the source actually answers the question. If the requested fact is '
              'missing return {"claims":[]}. Otherwise return {"claims":[{"source":1,"answer":'
              '"direct answer","quote":"short verbatim supporting words"}]}.')
    payload = {"sources":{str(i):doc.page_content for i,doc in enumerate(selected, 1)},
               "reader_hints":[text for _,text,_,_,_ in spans[:4]],
               "conversation":history, "question":question}
    messages = [("system", prompt), ("human", json.dumps(payload))]
    for _ in range(2):
        raw = model.invoke(messages).content
        try:
            claims = json.loads(raw)["claims"]
        except (KeyError, ValueError, TypeError):
            claims = None
        if claims == []:
            return []
        verified = cited_claims(selected, claims)
        if verified:
            return informative_claims(question, verified)
        messages.append(("human", "Return claims with a matching numeric source ID and a direct answer."))
    raise CitationError("The local model could not provide verifiable citations. Please try rephrasing.")

def cited_claims(docs, claims):
    verified = []
    sources = {str(i): doc for i, doc in enumerate(docs, 1)}
    for claim in claims if isinstance(claims, list) else []:
        if not isinstance(claim, dict):
            continue
        source = str(claim.get("source"))
        quote = claim.get("quote")
        statement = claim.get("answer")
        if source not in sources:
            continue
        if not isinstance(statement, str) or not statement.strip():
            continue
        doc = sources[source]
        # When the answer is itself a transcript excerpt, locate its words
        # directly, even if the model supplied the wrong passage number.
        if len(statement.split()) >= 7:
            literal = statement.strip().rstrip(".!?")
            matched_doc = next((d for d in docs if literal.casefold() in d.page_content.casefold()), None)
            if matched_doc is not None:
                doc, quote = matched_doc, literal
        if not isinstance(quote, str) or not quote.strip():
            continue
        pattern = r"\s+".join(re.escape(word) for word in quote.split())
        match = re.search(pattern, doc.page_content, re.I)
        if match:
            spans = [span for span in doc.metadata["offsets"] if span[0] < match.end() and span[1] > match.start()]
            verified.append((statement.strip(), match.group(), {"start": spans[0][2], "end": spans[-1][3]}))
    return verified
