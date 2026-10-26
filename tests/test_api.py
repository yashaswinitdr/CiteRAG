"""API tests with a fake pipeline, so they run without Ollama or any model."""

import httpx
import pytest
from fastapi.testclient import TestClient

import config
import main


class FakePipeline:
    def __init__(self, *args, **kwargs):
        pass

    def answer(self, question):
        if "boom" in question:
            raise httpx.ConnectError("ollama down")
        return {
            "answer": "SBERT takes about 5 seconds.",
            "sources": ["documents/reimers2019_sbert.pdf"],
            "passages": [
                {"source": "documents/reimers2019_sbert.pdf", "page": 0, "text": "about 5 seconds"}
            ],
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "RagPipeline", FakePipeline)
    monkeypatch.setattr(config, "API_KEY", None)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_sources_and_passages(client):
    response = client.post("/ask", json={"question": "How fast is SBERT?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "SBERT takes about 5 seconds."
    assert body["sources"] == ["documents/reimers2019_sbert.pdf"]
    assert body["passages"][0]["page"] == 0
    assert body["latency_ms"] >= 0


@pytest.mark.parametrize("question", ["", "  ", "hi", "x" * (config.MAX_QUESTION_CHARS + 1)])
def test_invalid_questions_are_rejected(client, question):
    assert client.post("/ask", json={"question": question}).status_code == 422


def test_missing_question_field_is_rejected(client):
    assert client.post("/ask", json={}).status_code == 422


def test_llm_outage_returns_503(client):
    response = client.post("/ask", json={"question": "please boom now"})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_api_key_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "secret")
    body = {"question": "How fast is SBERT?"}
    assert client.post("/ask", json=body).status_code == 401
    assert client.post("/ask", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/ask", json=body, headers={"X-API-Key": "secret"}).status_code == 200
