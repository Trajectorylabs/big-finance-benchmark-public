from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest
from trajectory import Client

from big_finance_harness.models.trajectory import TrajectoryClient
from big_finance_harness.types import Message, TextBlock, ToolSpec


@pytest.mark.asyncio
async def test_trajectory_client_records_model_calls_and_reward():
    sdk_client = Client(api_key="test")
    client = create_autospec(sdk_client, instance=True)
    sdk_client.close()
    client.trajectories.create.return_value.tid = "tid_test"
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_test",
                            function=SimpleNamespace(
                                name="final_answer",
                                arguments='{"answer": "42"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
        model="model_test",
        id="response_test",
        model_dump=Mock(return_value={"id": "response_test"}),
    )
    client.chat.completions.create.return_value = response
    adapter = TrajectoryClient(client)

    result = await adapter.chat(
        "system",
        [Message(role="user", content=[TextBlock(text="question")])],
        [
            ToolSpec(
                name="final_answer",
                description="Submit the answer.",
                input_schema={"type": "object"},
            )
        ],
    )

    assert result.tool_calls[0].input == {"answer": "42"}
    assert result.stop_reason == "tool_use"
    client.chat.completions.create.assert_called_once()
    assert client.chat.completions.create.call_args.kwargs["x_trajectory_id"] == "tid_test"

    adapter.log_reward(0.75, "3/4 rubric points")
    client.trajectories.log_reward.assert_called_once_with(
        "tid_test",
        reward_id="rubric",
        name="reward_accuracy",
        value=0.75,
        explanation="3/4 rubric points",
    )
    adapter.complete()
    client.trajectories.complete.assert_called_once_with(
        "tid_test",
        termination_reason="ENV_DONE",
    )
    client.close.assert_called_once_with()
