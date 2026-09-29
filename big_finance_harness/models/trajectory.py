from __future__ import annotations

import asyncio
import json

from trajectory import Client

from big_finance_harness.models.base import (
    ModelClient,
    ThinkingLevel,
    _to_oai_messages,
    _to_oai_tools,
)
from big_finance_harness.types import Message, ModelResponse, ToolSpec, ToolUseBlock


class TrajectoryClient(ModelClient):
    """Model client backed by one Trajectory session."""

    snapshot = "trajectory-session"

    def __init__(self, client: Client | None = None) -> None:
        self.client = client if client is not None else Client()
        self.tid = self.client.trajectories.create().tid

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float | None = None,
        thinking: ThinkingLevel = "off",
        max_output_tokens: int = 65536,
    ) -> ModelResponse:
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.snapshot,
            messages=_to_oai_messages(system, messages),
            x_trajectory_id=self.tid,
            max_tokens=max_output_tokens,
            extra_body={"tools": _to_oai_tools(tools), "tool_choice": "auto"},
        )
        choice = response.choices[0]
        message = choice.message
        tool_calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                arguments = {"_unparsed_arguments": call.function.arguments}
            tool_calls.append(
                ToolUseBlock(
                    id=call.id or "",
                    name=call.function.name,
                    input=arguments,
                )
            )
        stop_reason = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }.get(choice.finish_reason or "", "other")
        usage = response.usage
        return ModelResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            resolved_model=response.model,
            request_id=response.id,
            raw_response={"response": response.model_dump(mode="json")},
        )

    def log_reward(self, value: float, explanation: str) -> None:
        self.client.trajectories.log_reward(
            self.tid,
            reward_id="rubric",
            name="reward_accuracy",
            value=value,
            explanation=explanation,
        )

    def complete(self, termination_reason: str = "ENV_DONE") -> None:
        try:
            self.client.trajectories.complete(
                self.tid,
                termination_reason=termination_reason,
            )
        finally:
            self.client.close()
