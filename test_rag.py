import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.embeddings import DeterministicFakeEmbedding
from streamlit.testing.v1 import AppTest

import rag


class PipelineTests(unittest.TestCase):
    def test_factual_reader_cites_the_specific_caption(self):
        docs = rag.chunks([{"text":"The team needed film.", "start":2, "duration":2},
                           {"text":"Jim Lovell handed it to Anders.", "start":4, "duration":3}])
        start = docs[0].page_content.index("Jim Lovell")
        with patch("rag.read_spans", return_value=[(10, "Jim Lovell", docs[0], start, start+10)]):
            index = Mock()
            index.invoke.return_value = docs
            claims = rag.answer(index, "Who handed him the film?", [])
        self.assertEqual(claims, [("Jim Lovell", "Jim Lovell handed it to Anders.", {"start":4, "end":7})])

    def test_followup_uses_prior_answer_and_cites_selected_captions(self):
        rows = [{"text": "Lovell handed Anders the color film.", "start": 48, "duration": 6}]
        turns = [("Who helped with the photo?", [("Jim Lovell helped.", "Lovell", {"start":48,"end":54})])]
        index = Mock()
        index.invoke.return_value = rag.chunks(rows)
        with patch("rag.ChatOllama") as chat, patch("rag.read_spans", return_value=[]):
            chat.return_value.invoke.side_effect = [
                SimpleNamespace(content='{"query":"How did Jim Lovell help?"}'),

                SimpleNamespace(content=json.dumps({"claims":[{
                    "answer":"He handed Anders the color film.", "source":1, "quote":rows[0]["text"]}]})),
            ]
            claims = rag.answer(index, "What did he do?", rag.chat_context(turns), rows=rows)
            payload = json.loads(chat.return_value.invoke.call_args.args[0][1][1])
        self.assertEqual(payload["conversation"][0]["answer"], ["Jim Lovell helped."])
        self.assertEqual(claims[0][2], {"start":48, "end":54})
        index.invoke.assert_called_once_with("How did Jim Lovell help?")

    def test_long_video_retrieval_uses_rewritten_question(self):
        docs = rag.chunks([{"start":i*60,"duration":10,"text":("camera angle " * 60) + str(i)} for i in range(16)])
        model, index = Mock(), Mock()
        model.invoke.return_value.content = '{"query":"why did the crew change the camera angle?"}'
        index.invoke.return_value = [docs[7]]
        _, evidence = rag.evidence(index, "Why did they change it?",
                                [{"question":"What changed?", "answer":["The camera angle."]}],
                                "Camera setup", docs, model)
        index.invoke.assert_called_once_with("why did the crew change the camera angle?")
        self.assertIn(docs[7], evidence)

    def test_url_forms(self):
        for url in ["dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ?t=4",
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ"]:
            self.assertEqual(rag.video_id(url), "dQw4w9WgXcQ")
        with self.assertRaises(ValueError):
            rag.video_id("https://example.org/watch?v=dQw4w9WgXcQ")

    def test_source_validation(self):
        rows = [{"text": "Opening.", "start": 0, "duration": 2},
                {"text": "A caption.", "start": 3, "duration": 4}]
        good = {"answer": "The speaker says A caption.", "source": 1, "quote": "A caption."}
        claims = [good, dict(good, quote="Invented words."), dict(good, source=2)]
        self.assertEqual(rag.cited_claims(rag.chunks(rows), claims), [
            (good["answer"], good["quote"], {"start": 3, "end": 7})])

    def test_literal_answer_finds_its_caption_when_model_miscites_it(self):
        text = "The first workshop covers drawing and the second covers painting"
        docs = rag.chunks([{"text":text, "start":12, "duration":5},
                           {"text":"The final workshop ends here.", "start":100, "duration":4}])
        claim = {"answer":text, "source":2, "quote":"The final workshop ends here."}
        self.assertEqual(rag.cited_claims(docs, [claim]), [(text, text, {"start":12, "end":17})])

    @patch("youtube_transcript_api.YouTubeTranscriptApi.fetch")
    def test_video_chat(self, fetch):
        fetch.return_value.to_raw_data.return_value = [{"text": "The elephants eat grass.", "start": 4, "duration": 3}]
        with patch("rag.OllamaEmbeddings", return_value=DeterministicFakeEmbedding(size=16)), patch("rag.ChatOllama") as model, patch("rag.read_spans", return_value=[]):
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
