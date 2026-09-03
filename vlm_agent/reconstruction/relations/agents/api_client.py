from __future__ import annotations

import base64
import json
import http.client
import mimetypes
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_API_BASE = "https://api.openai.com/v1"


@dataclass
class APIConfig:
    model: str
    vision_model: str | None = None
    api_base: str | None = None
    api_key_env: str | None = "OPENAI_API_KEY"
    dry_run: bool = True
    image_detail: str | None = None
    default_api_base: str = DEFAULT_API_BASE
    resolved_api_base: str = field(init=False)
    api_base_source: str = field(init=False)
    resolved_api_key_env: str = field(init=False)

    def __post_init__(self) -> None:
        self.resolved_api_key_env = self.api_key_env or "OPENAI_API_KEY"
        if self.api_base:
            self.resolved_api_base = self.api_base
            self.api_base_source = "command_line"
        elif os.environ.get("OPEN_BASE_URL"):
            self.resolved_api_base = os.environ["OPEN_BASE_URL"]
            self.api_base_source = "OPEN_BASE_URL"
        elif os.environ.get("OPENAI_BASE_URL"):
            self.resolved_api_base = os.environ["OPENAI_BASE_URL"]
            self.api_base_source = "OPENAI_BASE_URL"
        elif os.environ.get("OPENAI_API_BASE"):
            self.resolved_api_base = os.environ["OPENAI_API_BASE"]
            self.api_base_source = "OPENAI_API_BASE"
        else:
            self.resolved_api_base = self.default_api_base
            self.api_base_source = "default"


class APIClientError(RuntimeError):
    def __init__(self, message: str, raw_response: str | None = None, retry_events: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.retry_events = retry_events or []


class SceneDeliveryAPIClient:
    def __init__(self, config: APIConfig):
        self.config = config
        self.retry_events: list[dict[str, Any]] = []
        self.last_retry_events: list[dict[str, Any]] = []
        self.last_response_model: str | None = None
        self.last_finish_reason: str | None = None

    @property
    def api_base(self) -> str:
        return self.config.resolved_api_base

    @property
    def api_base_source(self) -> str:
        return self.config.api_base_source

    @property
    def api_key_env(self) -> str:
        return self.config.resolved_api_key_env

    def require_api_key(self) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise APIClientError(f"API key missing: environment variable {self.api_key_env} is not set.")
        return api_key

    def usage_metadata(self) -> dict[str, Any]:
        """Return non-accounting response identity metadata."""

        return {
            "response_model": self.last_response_model,
            "finish_reason": self.last_finish_reason,
        }

    def call_json(
        self,
        role_name: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        use_vision_model: bool = False,
        max_retries: int = 3,
        retry_base_sleep: float = 2.0,
        image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        if self.config.dry_run:
            return self._dry_run_response(role_name, user_payload)

        api_key = self.require_api_key()
        model = self.config.vision_model if (use_vision_model or image_paths) and self.config.vision_model else self.config.model
        body = self.build_chat_completion_body(
            model=model,
            system_prompt=system_prompt,
            user_payload=user_payload,
            image_paths=image_paths,
        )
        request_body = json.dumps(body).encode("utf-8")
        delays = [2.0, 5.0, 10.0]
        retry_events: list[dict[str, Any]] = []
        self.last_retry_events = []
        self.last_response_model = None
        self.last_finish_reason = None
        last_raw_response: str | None = None
        total_attempts = max(1, int(max_retries) + 1)

        for attempt in range(1, total_attempts + 1):
            req = urllib.request.Request(
                self.api_base.rstrip("/") + "/chat/completions",
                data=request_body,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    raw_response = resp.read().decode("utf-8")
                data = json.loads(raw_response)
                choice = data["choices"][0]
                content = choice["message"]["content"]
                response_model = data.get("model")
                self.last_response_model = str(response_model) if response_model is not None else None
                finish_reason = choice.get("finish_reason")
                self.last_finish_reason = str(finish_reason) if finish_reason is not None else None
                last_raw_response = raw_response
                parsed = json.loads(content)
                if retry_events:
                    retry_events.append({
                        "role_name": role_name,
                        "model": model,
                        "attempt": attempt,
                        "status": "success_after_retry",
                    })
                    self.retry_events.extend(retry_events)
                    self.last_retry_events = retry_events
                return parsed
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_raw_response = detail
                error_message = f"{role_name} API call failed with HTTP {exc.code}: {detail}"
                event = self._retry_event(role_name, model, attempt, "HTTPError", error_message, max_retries)
            except urllib.error.URLError as exc:
                error_message = f"{role_name} API call failed: {exc}"
                event = self._retry_event(role_name, model, attempt, "URLError", error_message, max_retries)
            except http.client.RemoteDisconnected as exc:
                error_message = f"{role_name} API call disconnected without response: {exc}"
                event = self._retry_event(role_name, model, attempt, "RemoteDisconnected", error_message, max_retries)
            except (ConnectionResetError, TimeoutError) as exc:
                error_message = f"{role_name} API call connection failed: {exc}"
                event = self._retry_event(role_name, model, attempt, type(exc).__name__, error_message, max_retries)
            except json.JSONDecodeError as exc:
                error_message = f"{role_name} API call returned incomplete or non-JSON response: {exc}"
                event = self._retry_event(role_name, model, attempt, "JSONDecodeError", error_message, max_retries)
            except (KeyError, IndexError, TypeError) as exc:
                error_message = f"{role_name} API response did not match expected chat completion schema: {exc}"
                event = self._retry_event(role_name, model, attempt, type(exc).__name__, error_message, max_retries)

            retry_events.append(event)
            if attempt < total_attempts:
                sleep_seconds = delays[min(attempt - 1, len(delays) - 1)] if retry_base_sleep == 2.0 else retry_base_sleep
                retry_events[-1]["retry_sleep_seconds"] = sleep_seconds
                time.sleep(sleep_seconds)
                continue

            self.retry_events.extend(retry_events)
            self.last_retry_events = retry_events
            raise APIClientError(error_message, raw_response=last_raw_response, retry_events=retry_events)

        raise APIClientError(f"{role_name} API call failed unexpectedly.", retry_events=retry_events)


    def build_chat_completion_body(
        self,
        model: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        user_text = json.dumps(user_payload, ensure_ascii=False)
        if image_paths:
            content: str | list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            for image_path in image_paths:
                image_url = {"url": self._image_data_url(image_path)}
                if self.config.image_detail:
                    image_url["detail"] = self.config.image_detail
                content.append({"type": "image_url", "image_url": image_url})
        else:
            content = user_text
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_object"},
        }

    def _image_data_url(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise APIClientError(f"Vision image path does not exist: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _retry_event(
        self,
        role_name: str,
        model: str,
        attempt: int,
        exception_type: str,
        message: str,
        max_retries: int,
    ) -> dict[str, Any]:
        return {
            "role_name": role_name,
            "model": model,
            "attempt": attempt,
            "max_retries": max_retries,
            "exception_type": exception_type,
            "message": message,
            "will_retry": attempt <= max_retries,
        }

    def _dry_run_response(self, role_name: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        if role_name == "SceneDeliveryManager":
            return {
                "dry_run": True,
                "plan_id": "manager_plan_000",
                "objective": "Prepare scene delivery evidence tasks without calling the API.",
                "tasks": [
                    {
                        "task_id": "task_0001",
                        "agent_role": "DynamicTaskAgent",
                        "task_type": "inventory_and_schema_preview",
                        "needs_tool": False,
                        "policy_snippets": ["objects keep SAM3D ids", "relations allow attach and contain", "walls require top-down evidence"],
                        "instructions": "Summarize available evidence and identify missing final-output judgements.",
                    }
                ],
            }
        if role_name == "DynamicTaskAgent":
            return {
                "dry_run": True,
                "task_id": user_payload.get("task", {}).get("task_id", "task_0001"),
                "status": "mock_completed",
                "findings": [
                    "Object candidates were collected from current SAM3D/layout evidence.",
                    "No final wall, ceiling, or attach judgement was made in dry_run.",
                ],
                "proposed_state_updates": {},
            }
        if role_name == "ExecutionSupervisorAgent":
            return {
                "dry_run": True,
                "decision": "stop",
                "reason": "Dry run generated scaffolding and mock task logs; further API calls are disabled.",
                "ready_for_schema_guard": False,
            }
        if role_name == "DeliverySchemaGuard":
            return {
                "dry_run": True,
                "status": "skipped",
                "reason": "Dry run does not produce real final objects.json or relations.json.",
            }
        return {"dry_run": True, "status": "mock"}
