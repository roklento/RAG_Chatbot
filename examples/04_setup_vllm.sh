#!/bin/bash

# vLLM Setup Script for Qwen3-Next-80B-A3B-Instruct
# This script sets up and runs the vLLM server for efficient inference

set -e

echo "=================================================="
echo "vLLM Server Setup for RAG Chatbot"
echo "Model: Qwen3-Next-80B-A3B-Instruct (MoE)"
echo "=================================================="
echo ""

# Configuration
MODEL_NAME="Qwen/Qwen3-Next-80B-A3B-Instruct"
MODEL_PATH="${HF_MODEL_PATH:-/models/Qwen3-Next-80B-A3B-Instruct}"  # Use env var or default
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-262144}"  # 256K context
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"  # Adjust based on GPU count
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# Advanced options
ENABLE_MULTI_TOKEN_PREDICTION="${ENABLE_MULTI_TOKEN_PREDICTION:-true}"
DTYPE="${DTYPE:-auto}"

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Model Path: $MODEL_PATH"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Max Context: $MAX_MODEL_LEN tokens"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "  Multi-token Prediction: $ENABLE_MULTI_TOKEN_PREDICTION"
echo ""

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo "❌ vLLM not found. Installing..."
    pip install vllm>=0.6.0
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  Model not found at $MODEL_PATH"
    echo "Downloading from HuggingFace..."

    # Create directory
    mkdir -p "$(dirname "$MODEL_PATH")"

    # Download model using HuggingFace CLI
    huggingface-cli download "$MODEL_NAME" --local-dir "$MODEL_PATH"
fi

# Build vLLM command
CMD="vllm serve $MODEL_PATH"
CMD="$CMD --host $HOST"
CMD="$CMD --port $PORT"
CMD="$CMD --max-model-len $MAX_MODEL_LEN"
CMD="$CMD --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
CMD="$CMD --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
CMD="$CMD --dtype $DTYPE"
CMD="$CMD --trust-remote-code"

# Enable multi-token prediction for 2x faster streaming
if [ "$ENABLE_MULTI_TOKEN_PREDICTION" = "true" ]; then
    echo "✓ Enabling multi-token prediction (faster streaming)"
    # For Qwen models, use eagle speculative decoding if available
    # CMD="$CMD --speculative-config eagle"
fi

# Start server
echo ""
echo "Starting vLLM server..."
echo "Command: $CMD"
echo ""
echo "Server will be available at: http://$HOST:$PORT"
echo "OpenAI-compatible API: http://$HOST:$PORT/v1"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run vLLM server
eval $CMD
