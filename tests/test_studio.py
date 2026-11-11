import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import studio
from rag import excerpt_answer


class StudioTests(unittest.TestCase):
    def test_chat_history_is_scoped_to_video_and_caption_language(self):
        import sqlite3

        with closing(sqlite3.connect(":memory:")) as db, patch.object(studio, "semantic_index"), \
                patch.object(studio, "answer", return_value=[]) as generate:
            db.row_factory = sqlite3.Row
            db.executescript("""
                CREATE TABLE videos(id TEXT, lang TEXT, title TEXT, captions TEXT);
                CREATE TABLE turns(id INTEGER PRIMARY KEY, video TEXT, lang TEXT, question TEXT, claims TEXT);
                INSERT INTO videos VALUES('jNQXAC9IVRw','en','At the zoo','[]');
            """)
            for video, lang, question in [
                ("jNQXAC9IVRw", "en", "Who is speaking?"),
                ("jNQXAC9IVRw", "fr", "A question in another language"),
                ("dE-vOscpiNc", "en", "A question about another video"),
            ]:
                db.execute("INSERT INTO turns(video,lang,question,claims) VALUES(?,?,?,'[]')",
                           (video, lang, question))
            studio.process(db, "ask", {"video":"jNQXAC9IVRw", "lang":"en", "question":"What did he say?"})
            self.assertEqual(generate.call_args.args[2], [{"question":"Who is speaking?", "answer":[]}])

    def test_topic_question_is_distinct_from_a_specific_topic(self):
        rows = [
            {"text": "The topic of this video is what data engineers do every day.", "start": 4, "duration": 5},
            {"text": "They build pipelines and check the quality of incoming data.", "start": 90, "duration": 7},
            {"text": "Unity Catalog is a very important topic.", "start": 300, "duration": 3},
        ]
        title = "Daily work of data engineers"
        for question in ["topic of video?", "What's this video about?", "main subject"]:
            claims = excerpt_answer(rows, question, title)
            self.assertEqual(len(claims), 1)
            statement, quote, times = claims[0]
            self.assertIn(title, statement)
            self.assertIn("data engineers", quote)
            self.assertEqual(times, {"start": 4, "end": 9})
        self.assertEqual(excerpt_answer(rows, "Where is Unity Catalog mentioned?", title)[0][2]["start"], 300)

    def test_caption_continuation(self):
        rows = [
            {"text": "The planet is visible", "start": 8.2, "duration": 2},
            {"text": "through the side window.", "start": 10.2, "duration": 1.5},
        ]
        claims = excerpt_answer(rows, "Which window?")
        self.assertEqual(claims[0][2], {"start": 8, "end": 12})
        self.assertIn("planet is visible through", claims[0][0])
        self.assertEqual(excerpt_answer(rows, "sponsors"), [])

    def test_library_chat_and_bookmarks_persist(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(studio, "DATA", Path(folder)), \
             patch.object(studio.YouTubeTranscriptApi, "fetch") as fetch, patch.object(studio.requests, "get") as meta:
            fetch.return_value.to_raw_data.return_value = [{"text":"An orbital camera.", "start":5, "duration":3}]
            meta.return_value.json.return_value = {"title":"Orbit", "author_name":"Example"}
            video = {"video":"jNQXAC9IVRw", "lang":"en"}
            studio.initialize()
            with closing(studio.database()) as db, db:
                studio.process(db, "load", video)
                result = studio.process(db, "ask", {**video, "question":"Which camera?", "mode":"excerpt"})
                quote, timing = result["claims"][0][1:]
                for _ in range(2):
                    studio.process(db, "save", {**video, **timing, "quote":quote})
            with closing(studio.database()) as db, db:
                reopened = studio.process(db, "load", video)
                self.assertEqual(reopened["turns"][0]["claims"][0][2]["start"], 5)
                saved = studio.workspace(db)["moments"]
                self.assertEqual(len(saved), 1)
                self.assertEqual(studio.process(db, "unsave", {"id":saved[0]["id"]})["moments"], [])
            fetch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
