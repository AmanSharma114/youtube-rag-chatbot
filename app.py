import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import YouTubeTranscriptApiException

from rag import MISSING, answer, excerpt_answer, retriever, video_id


def stamp(seconds):
    return f"{seconds // 60}:{seconds % 60:02d}"


st.set_page_config(page_title="Transcript Chat", page_icon="▶")
st.title("Transcript Chat")
st.caption("Paste a YouTube link to ask questions about what the speaker said.")
with st.form("source"):
    link = st.text_input("YouTube video link", placeholder="https://www.youtube.com/watch?v=…")
    with st.expander("Chat settings"):
        language = st.text_input("Preferred caption language", "en")
        mode = st.radio("Answer mode", ["Transcript excerpts", "Local AI (Ollama)"],
                        help="Excerpt search works immediately. Local AI uses the two models in the setup guide.")
    load = st.form_submit_button("Chat with this video", type="primary")

if load:
    try:
        vid = video_id(link)
        with st.spinner("Extracting the video's transcript…"):
            rows = YouTubeTranscriptApi().fetch(vid, languages=[language.strip() or "en", "en"]).to_raw_data()
            if not any(row["text"].strip() for row in rows):
                raise ValueError("This transcript contains no spoken text.")
        new_index = None
        if mode == "Local AI (Ollama)":
            with st.spinner("Embedding the transcript locally…"):
                new_index = retriever(rows)
        st.session_state.update(index=new_index, vid=vid, rows=rows, history=[], mode=mode)
    except ValueError as exc:
        st.error(str(exc))
    except YouTubeTranscriptApiException as exc:
        st.error(f"Captions unavailable ({type(exc).__name__}). Try another language or video.")
    except Exception as exc:
        st.error(f"Indexing failed. Check the URL, connection and local Ollama models. {exc}")

if "rows" not in st.session_state:
    st.stop()

vid = st.session_state.vid
st.markdown(f"Chatting with [this YouTube video](https://youtu.be/{vid}) · {st.session_state.mode}")
with st.expander(f"Read transcript · {len(st.session_state.rows)} captions"):
    st.text("\n".join(f"{stamp(int(row['start']))}  {row['text']}" for row in st.session_state.rows))


def show_turn(question, claims):
    st.chat_message("user").text(question)
    with st.chat_message("assistant"):
        if not claims:
            st.write(MISSING)
        for statement, quote, meta in claims:
            st.text(statement)
            start = meta["start"]
            label = f'{stamp(start)}–{stamp(meta["end"])}'
            if quote != statement:
                st.text(quote)
            st.markdown(f"[{label} · Play passage](https://youtu.be/{vid}?t={start})")


history = st.session_state.history
for question, claims in history:
    show_turn(question, claims)
if question := st.chat_input("Ask about this transcript"):
    try:
        with st.spinner("Checking transcript evidence…"):
            if st.session_state.index is None:
                claims = excerpt_answer(st.session_state.rows, question)
            else:
                claims = answer(st.session_state.index, question, history[-1][0] if history else "")
        history.append((question, claims))
        show_turn(question, claims)
    except Exception as exc:
        st.error(f"Local generation failed. Check that Ollama is running and qwen2.5:3b is installed. {exc}")
