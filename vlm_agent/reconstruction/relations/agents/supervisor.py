from __future__ import annotations

from typing import Any

from relations.agents.api_client import SceneDeliveryAPIClient


def supervise_execution(client: SceneDeliveryAPIClient, system_prompt: str, scene_state: dict[str, Any], recent_results: list[dict[str, Any]]) -> dict[str, Any]:
    return client.call_json(
        "ExecutionSupervisorAgent",
        system_prompt,
        {"scene_state": scene_state, "recent_results": recent_results},
        use_vision_model=False,
    )
