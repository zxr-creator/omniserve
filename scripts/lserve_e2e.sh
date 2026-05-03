# model_name="Llama-3-8B-Instruct-Gradient-1048k"
attn_path=./attn_patterns/Llama-3-8B-Instruct-Gradient-1048k
model_path=./models/Llama-3-8B-Instruct-Gradient-1048k-w8a8-per-channel-kv8-per-tensor

# AIME24 prompts (already chat-templated for the Llama-8B family).
# Path is resolved relative to the omniserve repo root because that's
# where lserve_e2e_generation.py is launched from below.
dataset_path=./scripts/aime24_llama8b.jsonl


if [ ! -d "$model_path" ]; then
  mkdir -p models
  cd models
  git clone https://huggingface.co/mit-han-lab/Llama-3-8B-Instruct-Gradient-1048k-w8a8-per-channel-kv8-per-tensor
  cd ..
fi

precision="w8a8kv8"

NUM_RETRIEVAL_GPU_PAGE_BLOCKS=3000 \
NUM_STREAMING_GPU_PAGE_BLOCKS=200 \
python lserve_e2e_generation.py \
  --model $model_path \
  --ifb-mode \
  --precision $precision \
  --quant-path $model_path \
  --group-size -1 \
  --max-num-batched-tokens 4195000 \
  --max-num-seqs 1 \
  --omit-prompt \
  --kv-quant-granularity "per_tensor" \
  --chunk-prefill-size 32000 \
  --multiblock-switch 2048 \
  --static-sparse-attn-load-dir $attn_path \
  --static-sparsity 0.5 \
  --sparse-context-mode \
  --sparse-decode-mode 1 \
  --ctx-sink-token 128 \
  --ctx-local-token 8192 \
  --dec-sink-token 128 \
  --dec-local-token 256 \
  --sub-chunk-per-block 4 \
  --dynamic-sparse-token-budget 4096 \
  --selector-update-interval 4 \
  --dataset-path $dataset_path \
  --dataset-max-tokens 8192
