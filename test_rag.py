import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.embeddings import DeterministicFakeEmbedding
from streamlit.testing.v1 import AppTest

import rag


class PipelineTests(unittest.TestCase):
    def test_url_forms(self):
        for url in ["dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ?t=4",
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ"]:
            self.assertEqual(rag.video_id(url), "dQw4w9WgXcQ")
        with self.assertRaises(ValueError):
            rag.video_id("https://example.org/watch?v=dQw4w9WgXcQ")

    def test_source_validation(self):
        rows = [{"text": "Opening.", "start": 0, "duration": 2},
                {"text": "A caption.", "start": 3, "duration": 4}]
        index = Mock()
        index.invoke.return_value = rag.chunks(rows)
        good = {"answer": "The speaker says A caption.", "source": 1, "quote": "A caption."}
        claims = [good, dict(good, quote="Invented words."), dict(good, source=2)]
        with patch("rag.ChatOllama") as model:
            model.return_value.invoke.return_value = SimpleNamespace(content=json.dumps({"claims": claims}))
            self.assertEqual(rag.answer(index, "What was said?", ""), [
                (good["answer"], good["quote"], {"start": 3, "end": 7})])

    @patch("youtube_transcript_api.YouTubeTranscriptApi.fetch")
    def test_video_chat(self, fetch):
        fetch.return_value.to_raw_data.return_value = [{"text": "The elephants eat grass.", "start": 4, "duration": 3}]
        with patch("rag.OllamaEmbeddings", return_value=DeterministicFakeEmbedding(size=16)), patch("rag.ChatOllama") as model:
            model.return_value.invoke.return_value = SimpleNamespace(content='{"claims":[]}')
            app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=30).run()
            app.text_input[0].set_value("https://youtu.be/jNQXAC9IVRw")
            app.button[0].click().run()
            fetch.assert_called_once()
            app.chat_input[0].set_value("What do elephants eat?").run()
            self.assertEqual(app.session_state["history"][0][1][0][0], "The elephants eat grass.")
            app.radio[0].set_value("Local AI (Ollama)")
            app.button[0].click().run()
            self.assertIn("index", app.session_state)
            app.chat_input[0].set_value("Who sponsored the video?").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["history"], [("Who sponsored the video?", [])])


if __name__ == "__main__":
    unittest.main()
