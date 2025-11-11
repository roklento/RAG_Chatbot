#!/bin/bash

# Quick Start Script - Sets up everything for the RAG chatbot
# This script automates the entire setup process

set -e

echo "=================================================="
echo "RAG CHATBOT QUICK START"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0.32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check if Qdrant is running
echo "Step 1: Checking Qdrant..."
if ! docker ps | grep -q qdrant; then
    echo -e "${YELLOW}Qdrant not running. Starting Qdrant...${NC}"
    docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
    echo "Waiting for Qdrant to start..."
    sleep 5
    echo -e "${GREEN}✓ Qdrant started${NC}"
else
    echo -e "${GREEN}✓ Qdrant is already running${NC}"
fi
echo ""

# Step 2: Check environment file
echo "Step 2: Checking environment configuration..."
if [ ! -f "../.env" ]; then
    echo -e "${YELLOW}Creating .env from .env.example...${NC}"
    cp ../.env.example ../.env
    echo -e "${RED}⚠️  Please configure .env file with your settings${NC}"
    echo -e "${RED}   Especially: EMBEDDING_MODEL_PATH and VLLM_MODEL_PATH${NC}"
    echo ""
    read -p "Press Enter after configuring .env file..."
fi
echo -e "${GREEN}✓ Environment file exists${NC}"
echo ""

# Step 3: Setup Qdrant collections
echo "Step 3: Setting up Qdrant collections..."
python setup_qdrant_collections.py
echo ""

# Step 4: Ingest data
echo "Step 4: Ingesting data into Qdrant..."
echo -e "${YELLOW}This may take a few minutes...${NC}"
python ingest_data.py
echo ""

# Step 5: Check if vLLM model path is configured
echo "Step 5: Checking vLLM configuration..."
if grep -q "VLLM_MODEL_PATH=" ../.env && ! grep -q "VLLM_MODEL_PATH=$" ../.env; then
    echo -e "${GREEN}✓ vLLM model path configured${NC}"
    echo ""
    echo "=================================================="
    echo "SETUP COMPLETE!"
    echo "=================================================="
    echo ""
    echo "Next steps:"
    echo "1. Start vLLM server:"
    echo "   bash ../examples/04_setup_vllm.sh"
    echo ""
    echo "2. Test retrieval:"
    echo "   python ../examples/03_test_retrieval.py"
    echo ""
    echo "3. Run complete RAG chatbot:"
    echo "   python ../examples/08_complete_rag_demo.py --interactive"
    echo ""
else
    echo -e "${YELLOW}⚠️  vLLM model path not configured in .env${NC}"
    echo ""
    echo "=================================================="
    echo "DATA SETUP COMPLETE!"
    echo "=================================================="
    echo ""
    echo "You can now test retrieval:"
    echo "  python ../examples/03_test_retrieval.py"
    echo ""
    echo "To use the full RAG chatbot with generation:"
    echo "1. Configure vLLM model path in .env"
    echo "2. Start vLLM server: bash ../examples/04_setup_vllm.sh"
    echo "3. Run chatbot: python ../examples/08_complete_rag_demo.py --interactive"
    echo ""
fi
