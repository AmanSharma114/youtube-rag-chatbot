import json
from pathlib import Path

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import YouTubeTranscriptApiException

from rag import MISSING, answer, retriever, video_id


def stamp(seconds):
    return f"{seconds // 60}:{seconds % 60:02d}"


st.set_page_config(page_title="Transcript Chat", page_icon="▶")
st.title("Transcript Chat")
st.caption("Ask a question and jump to the passage that supports the answer.")
with st.sidebar.form("source"):
    st.header("Transcript source")
    link = st.text_input("YouTube URL", placeholder="Leave empty for the sample lesson")
    language = st.text_input("Preferred caption language", "en")
    load = st.form_submit_button("Index transcript")

if load:
    try:
        vid = video_id(link) if link.strip() else None
        with st.spinner("Reading captions and embedding passages locally…"):
            if vid:
                rows = YouTubeTranscriptApi().fetch(vid, languages=[language.strip(), "en"]).to_raw_data()
            else:
                rows = json.loads(Path(__file__).with_name("sample.json").read_text())
            new_index = retriever(rows)
        st.session_state.update(index=new_index, vid=vid, history=[])
    except YouTubeTranscriptApiException as exc:
        st.error(f"Captions unavailable ({type(exc).__name__}). Try another language or video.")
    except Exception as exc:
        st.error(f"Indexing failed. Check the URL, connection and local Ollama models. {exc}")

if "index" not in st.session_state:
    st.info("Choose Index transcript to start. An empty URL opens the sample lesson on retrieval.")
    st.stop()

vid = st.session_state.vid
st.caption(f"Active transcript · {vid or 'Sample lesson on retrieval'}")


def show_turn(question, claims):
    st.chat_message("user").text(question)
    with st.chat_message("assistant"):
        if not claims:
            st.write(MISSING)
        for statement, quote, meta in claims:
            st.text(statement)
            start = meta["start"]
            label = f'{stamp(start)}–{stamp(meta["end"])}'
            with st.expander(f"Transcript evidence · {label}"):
                st.text(quote)
                if vid:
                    st.link_button(f"Play from {stamp(start)}", f"https://youtu.be/{vid}?t={start}")


history = st.session_state.history
for question, claims in history:
    show_turn(question, claims)
if question := st.chat_input("Ask about this transcript"):
    try:
        with st.spinner("Checking transcript evidence…"):
            claims = answer(st.session_state.index, question, history[-1][0] if history else "")
        history.append((question, claims))
        show_turn(question, claims)
    except Exception as exc:
        st.error(f"Local generation failed. Check that Ollama is running and qwen2.5:3b is installed. {exc}")
