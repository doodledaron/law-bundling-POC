"""
Minimal OpenRouter client that mimics the parts of the google-genai SDK
used by TextBasedProcessor and RelevanceProcessor, so those classes only
need their imports changed.

Supported surface:
  - client.models.generate_content(model=..., contents=[str | Part, ...], config=GenerateContentConfig)
  - types.Part.from_bytes(data=..., mime_type=...)
  - types.GenerateContentConfig(temperature=..., top_p=..., max_output_tokens=...)
  - response.text, response.candidates, response.usage_metadata
"""
import base64
import logging
from types import SimpleNamespace

import requests

from config import Config

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 180


class Part:
    """Holds raw bytes plus a mime type, like google.genai.types.Part."""

    def __init__(self, data, mime_type):
        self.data = data
        self.mime_type = mime_type

    @classmethod
    def from_bytes(cls, data, mime_type):
        return cls(data, mime_type)


class GenerateContentConfig(dict):
    """Keyword settings for a request; keys follow the Gemini naming."""

    def __init__(self, temperature=None, top_p=None, top_k=None, max_output_tokens=None, **_):
        # top_k is not part of the OpenRouter chat API, so it is dropped here
        super().__init__()
        if temperature is not None:
            self["temperature"] = temperature
        if top_p is not None:
            self["top_p"] = top_p
        if max_output_tokens is not None:
            self["max_tokens"] = max_output_tokens


class _Models:
    def __init__(self, api_key):
        self._api_key = api_key

    def generate_content(self, model, contents, config=None):
        messages = [{"role": "user", "content": _to_openrouter_content(contents)}]
        body = {"model": model, "messages": messages}
        if config:
            body.update(config)

        resp = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=body,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        # Keep the HTTP status in the message so the caller's retry logic can see it
        resp.raise_for_status()
        return _to_response(resp.json())


def _to_openrouter_content(contents):
    parts = []
    for item in contents:
        if isinstance(item, Part):
            encoded = base64.b64encode(item.data).decode()
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{item.mime_type};base64,{encoded}"},
            })
        else:
            parts.append({"type": "text", "text": str(item)})
    return parts


def _to_response(payload):
    choice = payload["choices"][0]
    # Reasoning models can return null content; treat it as empty text
    text = (choice.get("message", {}).get("content") or "").strip()
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]))] if text else [],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
            total_token_count=usage.get("total_tokens", prompt_tokens + completion_tokens),
        ),
    )


class Client:
    def __init__(self, api_key=None, **_):
        self.models = _Models(api_key or Config.OPENROUTER_API_KEY)


# Stand-ins for `from google import genai` / `from google.genai import types`
genai = SimpleNamespace(Client=Client)
types = SimpleNamespace(Part=Part, GenerateContentConfig=GenerateContentConfig)
