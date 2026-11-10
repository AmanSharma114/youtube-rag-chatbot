"""Local HTTP interface for the Cueframe video workspace."""
import json
import os
import sqlite3
from contextlib import closing
from functools import lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import YouTubeTranscriptApiException

from rag import answer, excerpt_answer, overview_question, retriever, video_id

ROOT = Path(__file__).resolve().parent
DATA = Path(os.getenv("CUEFRAME_DATA_DIR", Path.home() / ".cueframe"))
WEB = ROOT / "web"


def database():
    DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATA / "workspace.sqlite", timeout=10)
    db.row_factory = sqlite3.Row
    return db


def initialize():
    with closing(database()) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS videos(id TEXT, lang TEXT, title TEXT, author TEXT,
          captions TEXT, added INTEGER DEFAULT (unixepoch()), PRIMARY KEY(id,lang));
        CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY, video TEXT, lang TEXT,
          question TEXT, claims TEXT);
        CREATE TABLE IF NOT EXISTS moments(id INTEGER PRIMARY KEY, video TEXT, lang TEXT,
          start INTEGER, end INTEGER, quote TEXT, UNIQUE(video,lang,start,end,quote));
        """)


@lru_cache(maxsize=3)
def semantic_index(vid, lang):
    with closing(database()) as db:
        rows = db.execute("SELECT captions FROM videos WHERE id=? AND lang=?", (vid, lang)).fetchone()
    return retriever(json.loads(rows["captions"]))


def workspace(db):
    videos = db.execute("SELECT id,lang,title,author FROM videos ORDER BY added DESC,rowid DESC")
    moments = db.execute("SELECT m.*, v.title FROM moments m JOIN videos v ON v.id=m.video AND v.lang=m.lang ORDER BY m.id DESC")
    return {"videos": list(map(dict, videos)), "moments": list(map(dict, moments))}


def process(db, route, data):
    if route == "workspace":
        return workspace(db)
    if route == "unsave":
        db.execute("DELETE FROM moments WHERE id=?", (int(data["id"]),))
        return workspace(db)
    vid = video_id(data["video"])
    lang = data.get("lang", "en").strip() or "en"
    key = (vid, lang)
    record = db.execute("SELECT * FROM videos WHERE id=? AND lang=?", key).fetchone()
    if route == "load" and not record:
        fetched = YouTubeTranscriptApi().fetch(vid, languages=list(dict.fromkeys([lang, "en"])))
        rows = [row for row in fetched.to_raw_data() if row["text"].strip()]
        if not rows:
            raise ValueError("This video has no readable captions.")
        title, author = f"YouTube video {vid}", "YouTube"
        try:
            response = requests.get("https://www.youtube.com/oembed", params={"url": f"https://youtu.be/{vid}", "format":"json"}, timeout=5)
            response.raise_for_status()
            meta = response.json()
            title, author = meta["title"], meta["author_name"]
        except (requests.RequestException, KeyError, ValueError):
            pass
        db.execute("INSERT OR IGNORE INTO videos(id,lang,title,author,captions) VALUES(?,?,?,?,?)",
                   (*key, title, author, json.dumps(rows)))
        record = db.execute("SELECT * FROM videos WHERE id=? AND lang=?", key).fetchone()
    if not record:
        raise ValueError("Open this video first.")
    rows = json.loads(record["captions"])
    if route == "load":
        turns = db.execute("SELECT question, claims FROM turns WHERE video=? AND lang=? ORDER BY id", key)
        return {**dict(record), "captions": rows, "turns": [{"question": t[0], "claims": json.loads(t[1])} for t in turns]}
    if route == "ask":
        question = data["question"].strip()
        if not 1 <= len(question) <= 2000:
            raise ValueError("Enter a question of up to 2,000 characters.")
        if data.get("mode") == "ai" and not overview_question(question):
            prev = db.execute("SELECT question FROM turns WHERE video=? AND lang=? ORDER BY id DESC LIMIT 1", key).fetchone()
            claims = answer(semantic_index(*key), question, prev[0] if prev else "")
        else:
            claims = excerpt_answer(rows, question, record["title"])
        db.execute("INSERT INTO turns(video,lang,question,claims) VALUES(?,?,?,?)", (*key, question, json.dumps(claims)))
        return {"question":question, "claims":claims}
    if route == "save":
        db.execute("INSERT OR IGNORE INTO moments(video,lang,start,end,quote) VALUES(?,?,?,?,?)",
                   (*key, int(data["start"]), int(data["end"]), data["quote"]))
        return workspace(db)
    raise ValueError("Unknown operation.")


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if urlsplit(self.path).path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        origin = self.headers.get("Origin")
        if (self.headers.get("Host") not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
                or (origin and origin != "http://" + self.headers["Host"])
                or self.headers.get_content_type() != "application/json" or not self.path.startswith("/api/")):
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 100000:
                raise ValueError("Request is too large or empty.")
            data = json.loads(self.rfile.read(length))
            with closing(database()) as db, db:
                result = process(db, self.path[5:], data)
            status = 200
        except YouTubeTranscriptApiException:
            status, result = 422, {"error":"YouTube could not return captions. Check the caption language or try another captioned video."}
        except (KeyError, ValueError, TypeError) as exc:
            status, result = 400, {"error":str(exc)}
        except Exception:
            status, result = 503, {"error":"The request could not finish. Check your connection. For local AI, verify Ollama and both models are installed."}
        body = json.dumps(result).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control","no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    initialize()
    port = int(os.getenv("CUEFRAME_PORT", "8501"))
    print(f"Cueframe is running at http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(WEB))).serve_forever()
