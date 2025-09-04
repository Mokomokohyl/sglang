#!/bin/bash
rm -rf /tmp/kv_dumps
python -m sglang.bench_one_batch --model-path /mnt/model/llama-2-7b-hf --correct --batch-size 3 --max-total-tokens 2048 --disable-cuda-graph --attention-backend clusterfusion > run.log
python -m sglang.bench_one_batch --model-path /mnt/model/llama-2-7b-hf --correct --batch-size 3 --max-total-tokens 2048 --disable-cuda-graph --attention-backend flashinfer > run_ref.log
python3 compare_kv.py > compare.log 2>&1
