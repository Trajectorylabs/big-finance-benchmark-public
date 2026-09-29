from ingest_trajectory import build_benchmark


def test_public_subset_ingestion_contains_only_test_tasks():
    benchmark = build_benchmark()

    assert benchmark.name == "big-finance-public-subset"
    assert len(benchmark.tasks) == 50
    assert {task.split for task in benchmark.tasks} == {"test"}
    assert all("big_finance_subset.jsonl" in task.run_command for task in benchmark.tasks)
