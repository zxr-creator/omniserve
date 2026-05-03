# model_name="DeepSeek-R1-Distill-Llama-8B"

model_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
precision="w16a16kv8"
dataset_path=./aime24_llama8b.jsonl
output_dir=/root/omniserve/omniserve/results

dynamic_sparse_token_budgets=(464 976 1488 2000 2512 3024)


for dynamic_sparse_token_budget in "${dynamic_sparse_token_budgets[@]}"; do
  echo "Running with dynamic_sparse_token_budget=${dynamic_sparse_token_budget}"

  python ../lserve_e2e_generation.py \
    --model $model_path \
    --ifb-mode \
    --tokenizer-mode "auto" \
    --load-format "auto" \
    --dtype "auto" \
    --block-size 16 \
    --seed 0 \
    --swap-space 4 \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 262144 \
    --max-num-seqs 256 \
    --max-paddings 256 \
    --precision $precision \
    --group-size -1 \
    --kv-quant-granularity "per_tensor" \
    --chunk-prefill-size 32000 \
    --static-sparsity 0.0 \
    --ctx-sink-token 128 \
    --ctx-local-token 8192 \
    --dec-sink-token 16 \
    --dec-local-token 32 \
    --sparse-decode-mode 1 \
    --sub-chunk-per-block 1 \
    --dynamic-sparse-token-budget ${dynamic_sparse_token_budget} \
    --selector-update-interval 1 \
    --multiblock-switch 2048 \
    --output-path ${output_dir}/lserve_aime24_budget${dynamic_sparse_token_budget}.jsonl
done