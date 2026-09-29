# /// script
# dependencies = ["trajectory-sdk==0.7.21"]
# ///
import json
import os
from pathlib import Path

from trajectory import BenchmarkSpec, Client, SecretRef, TaskSpec
from trajectory.lib import DockerfileBuild, push, wait_for_benchmark_images
from trajectory.types.benchmarks.task_spec import EnvResources

ROOT = Path(__file__).parent
DATASET = ROOT / "data" / "big_finance_subset.jsonl"


def build_benchmark() -> BenchmarkSpec:
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line]
    return BenchmarkSpec(
        name="big-finance-public-subset",
        runtime=DockerfileBuild("trajectory.Dockerfile"),
        tasks=[
            TaskSpec(
                name=f"big-finance/{row['id']}",
                split="test",
                run_command=(
                    "python -u /opt/big_finance/scripts/run_eval_set.py --trajectory "
                    "--dataset /opt/big_finance/data/big_finance_subset.jsonl "
                    "--run-id trajectory --max-steps 30 --judge openai:gpt-5.4-mini"
                ),
                env_vars={
                    "BFB_TASK_ID": row["id"],
                    "OPENAI_API_KEY": SecretRef(secret_ref="OPENAI_API_KEY"),
                    "TAVILY_API_KEY": SecretRef(secret_ref="TAVILY_API_KEY"),
                    "SEC_EDGAR_USER_AGENT": SecretRef(
                        secret_ref="SEC_EDGAR_USER_AGENT"
                    ),
                },
                env_resources=EnvResources(network_mode="public"),
            )
            for row in rows
        ],
    )


def main() -> None:
    client = Client()
    result = push(
        client,
        build_benchmark(),
        agent_id=os.environ["TRAJECTORY_AGENT_ID"],
        root=ROOT,
    )
    wait_for_benchmark_images(client, result.bench_id)
    print(result.bench_id)


if __name__ == "__main__":
    main()
