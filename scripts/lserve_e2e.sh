# model_name="DeepSeek-R1-Distill-Llama-8B"

model_path="/tmp/models/DeepSeek-R1-Distill-Llama-8B"
precision="w16a16kv8"
dataset_path=./aime24_llama8b.jsonl
mkdir -p ../results
output_dir=../results

export NUM_RETRIEVAL_GPU_PAGE_BLOCKS=3000
export NUM_STREAMING_GPU_PAGE_BLOCKS=200

dynamic_sparse_token_budgets=(1856)

for dynamic_sparse_token_budget in "${dynamic_sparse_token_budgets[@]}"; do
  echo "Running with dynamic_sparse_token_budget=${dynamic_sparse_token_budget}"

  python ../lserve_e2e_generation.py  \
  --model $model_path \
  --ifb-mode \
  --precision $precision \
  --group-size -1 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 262144 \
  --max-num-seqs 1 \
  --omit-prompt \
  --kv-quant-granularity "per_tensor" \
  --chunk-prefill-size 32000 \
  --multiblock-switch 2048 \
  --sparse-decode-mode 1 \
  --ctx-sink-token 128 \
  --ctx-local-token 8192 \
  --dec-sink-token 64 \
  --dec-local-token 128 \
  --sub-chunk-per-block 4 \
  --dynamic-sparse-token-budget ${dynamic_sparse_token_budget} \
  --selector-update-interval 1 \
  --dataset-max-tokens 28672 \
  --dataset-path "${dataset_path}" \
  --output-path ${output_dir}/lserve_aime24_budget${dynamic_sparse_token_budget}.jsonl   2>&1 | tee ${output_dir}/lserve_aime24_budget${dynamic_sparse_token_budget}.log
done