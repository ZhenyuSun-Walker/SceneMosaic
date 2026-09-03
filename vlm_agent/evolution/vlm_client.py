from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from openai import OpenAI


class VLMClient:
    def __init__(
        self,
        url: str | None,
        model: str,
        api_key: str,
        timeout_s: float = 60.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self.model = model
        self.timeout_s = timeout_s
        self.headers = headers
        client_kwargs: dict[str, object] = {
            "api_key": api_key,
            "timeout": timeout_s,
        }
        if url:
            client_kwargs["base_url"] = url
        if headers:
            client_kwargs["default_headers"] = headers
        self.client = OpenAI(**client_kwargs)
        self.last_usage: dict[str, int] | None = None

    def plan(self, prompt: str, images: Sequence[str] | None = None) -> str:
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        for data_url in self._encode_images(images):
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
        )
        self.last_usage = self._extract_usage(response)
        return self._extract_response_text(response)

    def extract_json_payload(self, text: str) -> dict:
        return self._extract_json_payload(text)

    @staticmethod
    def _encode_images(images: Sequence[str] | None) -> list[str]:
        if not images:
            return []
        encoded: list[str] = []
        for image in images:
            image_path = Path(image)
            if not image_path.exists():
                continue
            mime_type, _ = mimetypes.guess_type(image_path.name)
            if mime_type is None:
                mime_type = "application/octet-stream"
            data = base64.b64encode(image_path.read_bytes()).decode("ascii")
            encoded.append(f"data:{mime_type};base64,{data}")
        return encoded

    @staticmethod
    def _extract_response_text(response: object) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text

        choices = getattr(response, "choices", None)
        if choices:
            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    return content

        if hasattr(response, "model_dump"):
            return json.dumps(response.model_dump(), ensure_ascii=False)
        return str(response)

    @staticmethod
    def _extract_usage(response: object) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is not None:
            if hasattr(usage, "model_dump"):
                usage_dict = usage.model_dump()
            elif isinstance(usage, dict):
                usage_dict = usage
            else:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            normalized_usage: dict[str, int] = {}
            for key, value in usage_dict.items():
                if value is None:
                    continue
                if isinstance(value, dict):
                    nested_value = value.get("value")
                    if nested_value is None:
                        nested_value = value.get("total")
                    if nested_value is None:
                        continue
                    value = nested_value
                try:
                    normalized_usage[key] = int(value)
                except (TypeError, ValueError):
                    continue
            return normalized_usage

        if hasattr(response, "model_dump"):
            response_dict = response.model_dump()
            usage_dict = response_dict.get("usage")
            if isinstance(usage_dict, dict):
                return {
                    key: int(value)
                    for key, value in usage_dict.items()
                    if value is not None
                }

        return {}

    @staticmethod
    def _extract_json_payload(text: str) -> dict:
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced_match:
            candidate = fenced_match.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        start = text.find("{")
        if start != -1:
            depth = 0
            for idx in range(start, len(text)):
                char = text[idx]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:idx + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break

        return {"raw_response": text}


@dataclass(frozen=True)
class DummyVLMClient:
    default_action: str = "{\"actions\": []}"

    def plan(self, prompt: str, images: Sequence[str] | None = None) -> str:
        del prompt, images
        return self.default_action

    def extract_json_payload(self, text: str) -> dict:
        return VLMClient._extract_json_payload(text)


def parse_vlm_json(text: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"actions": [], "raw_response": text}
    if not isinstance(payload, dict):
        return {"actions": [], "raw_response": text}
    payload.setdefault("actions", [])
    return payload
