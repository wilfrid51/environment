#!/bin/bash
# vLLM Inference Server Startup Script
#
# This script starts a vLLM server that provides an OpenAI-compatible API
# endpoint for LLM inference. The server can be used locally for evaluation
# and testing.
#
# Configuration:
#   HUGGINGFACE_MODEL_NAME: HuggingFace model identifier
#   PORT: Port number for the API server (default: 20000)
#   MAX_MODEL_LEN: Maximum context length for the model
#   VLLM_USE_FLASHINFER_SAMPLER: Disable FlashInfer sampler (set to 0)
#   CUDA_VISIBLE_DEVICES: GPU device to use (0 = first GPU)
#
# Usage:
#   bash inference.sh
#
# The server will be accessible at http://localhost:20000/v1

# Model configuration
export HUGGINGFACE_MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
export PORT="20000"
export MAX_MODEL_LEN="40960"

# Disable FlashInfer sampler (may cause issues on some systems)
export VLLM_USE_FLASHINFER_SAMPLER=0

# Start vLLM server with OpenAI-compatible API
# Using GPU 0, binding to all interfaces (0.0.0.0) for remote access
# GPU memory utilization set to 95% for maximum throughput
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model ${HUGGINGFACE_MODEL_NAME} \
    --port ${PORT} \
    --max-model-len ${MAX_MODEL_LEN} \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.95
