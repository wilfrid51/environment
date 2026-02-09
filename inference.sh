export HUGGINGFACE_MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
export PORT="20000"
export MAX_MODEL_LEN="40960"

export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model ${HUGGINGFACE_MODEL_NAME} \
    --port ${PORT} \
    --max-model-len ${MAX_MODEL_LEN} \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.95
