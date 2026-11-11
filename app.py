from html import escape
from math import sin
from pathlib import Path

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import YouTubeTranscriptApiException

from rag import MISSING, answer, chat_context, excerpt_answer, retriever, video_id


def stamp(seconds):
    if seconds >= 3600:
        return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60}:{seconds % 60:02d}"


st.set_page_config(page_title="Cueframe · Video chat", page_icon="▶", layout="wide",
                   initial_sidebar_state="collapsed")
st.html("<style>" + Path(__file__).with_name("style.css").read_text(encoding="utf-8") + "</style>")
st.html('<div class="brand"><i aria-hidden="true">▷</i> Cueframe</div>')
if "rows" not in st.session_state:
    bars = "".join(f'<b style="height:{12 + int(84 * abs(sin(i*.23)*sin(i*1.17)))}px"></b>'
                   for i in range(54))
    st.html(f'<div class="hero"><div class="intro"><h1>Chat with your<br><em>YouTube video.</em></h1>'
            '<p>Ask about a video and jump straight to the moment that answers your question.</p>'
            f'</div><div aria-hidden="true" class="scope">{bars}</div></div>')
with st.form("source"):
    link = st.text_input("YouTube video link", placeholder="Paste a YouTube link…")
    with st.expander("Chat settings"):
        language = st.text_input("Preferred caption language", "en")
        mode = st.radio("Answer mode", ["Transcript excerpts", "Local AI (Ollama)"],
                        help="Excerpt search works immediately. Local AI uses the two models in the setup guide.")
    load = st.form_submit_button("Chat with this video", type="primary")

example = False
if "rows" not in st.session_state:
    example = st.button("Try a video · Me at the zoo", icon=":material/play_circle:")
if load or example:
    try:
        vid = "jNQXAC9IVRw" if example else video_id(link)
        with st.spinner("Extracting the video's transcript…"):
            rows = YouTubeTranscriptApi().fetch(vid, languages=[language.strip() or "en", "en"]).to_raw_data()
            if not any(row["text"].strip() for row in rows):
                raise ValueError("This transcript contains no spoken text.")
        new_index = None
        if mode == "Local AI (Ollama)":
            with st.spinner("Embedding the transcript locally…"):
                new_index = retriever(rows)
        st.session_state.update(index=new_index, vid=vid, rows=rows, history=[], mode=mode)
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))
    except YouTubeTranscriptApiException as exc:
        st.error(f"Captions unavailable ({type(exc).__name__}). Try another language or video.")
    except Exception as exc:
        st.error(f"Indexing failed. Check the URL, connection and local Ollama models. {exc}")

if "rows" not in st.session_state:
    st.stop()

vid = st.session_state.vid
with st.expander(f"Video & transcript · {len(st.session_state.rows)} captions"):
    player, transcript = st.columns([1, 1])
    player.video(f"https://www.youtube.com/watch?v={vid}")
    with transcript.container(height=240):
        st.text("\n".join(f"{stamp(int(row['start']))}  {row['text']}" for row in st.session_state.rows))
st.subheader("Video conversation")
st.caption(f"{st.session_state.mode} · Select a timestamp to play the cited moment on YouTube.")


def show_turn(question, claims):
    st.chat_message("user").text(question)
    with st.chat_message("assistant"):
        if not claims:
            st.write(MISSING)
        for statement, quote, meta in claims:
            start = meta["start"]
            label = f'{stamp(start)}–{stamp(meta["end"])}'
            evidence = f'<blockquote class="evidence">{escape(quote)}</blockquote>' if quote != statement else ""
            st.html(f'<p class="answer">{escape(statement)}</p>{evidence}'
                    f'<a class="cue" href="https://youtu.be/{vid}?t={start}" target="_blank" '
                    f'rel="noopener noreferrer" aria-label="Play passage {label}">▶ {label}</a>')


history = st.session_state.history
for question, claims in history:
    show_turn(question, claims)
if question := st.chat_input("Ask about this transcript"):
    try:
        with st.spinner("Checking transcript evidence…"):
            if st.session_state.index is None:
                claims = excerpt_answer(st.session_state.rows, question)
            else:
                claims = answer(st.session_state.index, question, chat_context(history), rows=st.session_state.rows)
        history.append((question, claims))
        show_turn(question, claims)
    except Exception as exc:
        st.error(f"Local generation failed. Check that Ollama is running and qwen2.5:3b is installed. {exc}")
