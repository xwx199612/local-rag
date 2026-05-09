import requests

from app.config import LLM_MODEL

OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"


def ask_llm(prompt: str, model: str = LLM_MODEL) -> str:
    try:
        res = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )
        res.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Start Ollama and make sure it is listening on 127.0.0.1:11434."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama request failed: {res.text}") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("Ollama timed out while generating the answer.") from exc

    return res.json()["response"]
