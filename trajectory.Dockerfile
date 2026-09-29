FROM python:3.12-slim

WORKDIR /opt/big_finance
COPY pyproject.toml README.md ./
COPY big_finance_harness big_finance_harness
COPY scripts/run_eval_set.py scripts/run_eval_set.py
COPY data/big_finance_subset.jsonl data/big_finance_subset.jsonl
RUN python -m pip install --no-cache-dir ".[trajectory]"
CMD ["sleep", "infinity"]
