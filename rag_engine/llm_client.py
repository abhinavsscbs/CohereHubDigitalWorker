import json
from typing import Any, Dict, List, Optional

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import (
    COHERE_API_KEY,
    COHERE_ENDPOINT_URL,
    COHERE_REQUEST_TIMEOUT_SEC,
    COHERE_VERIFY_SSL,
    LLM_MAX_TOKENS,
    LLM_MODELS,
    LLM_TEMPERATURES,
)


def _messages_to_prompt(messages: List[BaseMessage]) -> str:
    parts = []
    for m in messages:
        role = "User"
        if m.type == "system":
            role = "System"
        elif m.type == "ai":
            role = "Assistant"
        parts.append(f"{role}: {m.content}")
    return "\n".join(parts).strip()


def _extract_text(resp_json: Dict[str, Any]) -> str:
    if isinstance(resp_json, dict):
        for key in ("response", "text", "output"):
            if isinstance(resp_json.get(key), str):
                return resp_json[key]
        msg = resp_json.get("message")
        if isinstance(msg, str):
            return msg
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list) and content:
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return str(part.get("text", ""))
        if "generations" in resp_json and isinstance(resp_json["generations"], list):
            first = resp_json["generations"][0] if resp_json["generations"] else {}
            if isinstance(first, dict) and "text" in first:
                return str(first.get("text", ""))
    return ""


class CohereEndpointChatModel(BaseChatModel):
    def __init__(
        self,
        model_name: str,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        verify_ssl: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.endpoint_url = endpoint_url or COHERE_ENDPOINT_URL
        self.api_key = api_key or COHERE_API_KEY
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verify_ssl = COHERE_VERIFY_SSL if verify_ssl is None else verify_ssl

    @property
    def _llm_type(self) -> str:
        return "cohere_endpoint"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.endpoint_url or "YOUR_COHERE_ENDPOINT" in self.endpoint_url:
            raise RuntimeError("COHERE_ENDPOINT_URL is not configured.")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "message": _messages_to_prompt(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "text"},
        }
        if self.model_name:
            payload["model"] = self.model_name
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)

        resp = requests.post(
            self.endpoint_url,
            headers=headers,
            json=payload,
            timeout=COHERE_REQUEST_TIMEOUT_SEC,
            verify=self.verify_ssl,
        )
        resp.raise_for_status()
        text = _extract_text(resp.json()) or ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def get_llm(key: str) -> CohereEndpointChatModel:
    model_name = LLM_MODELS.get(key, "")
    temperature = LLM_TEMPERATURES.get(key, 0.0)
    max_tokens = LLM_MAX_TOKENS.get(key, 2048)
    return CohereEndpointChatModel(
        model_name=model_name,
        endpoint_url=COHERE_ENDPOINT_URL,
        api_key=COHERE_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )
