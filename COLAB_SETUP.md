# Google Colab Setup Guide

Complete guide for running the Rise Online RAG Chatbot in Google Colab with Gemma-3n-E4B-it.

---

## 🎯 Overview

This guide shows you how to run the complete RAG chatbot in Google Colab using:
- **Gemma-3n-E4B-it** for generation and query processing
- **Qwen3-Embedding-4B** for embeddings (unchanged)
- **Qwen3-Reranker-4B** for reranking (unchanged)
- **In-memory Qdrant** for vector storage
- **32K context window** (Gemma-3n capability)

---

## 📋 Prerequisites

### Google Colab Requirements
- Google Colab account (free or Pro)
- **GPU runtime** (T4, A100, or V100)
- Recommended: Colab Pro for better GPU access

### Local Requirements (for uploading files)
- Git repository cloned locally
- All project files ready

---

## 🚀 Quick Start (Colab Notebook Cells)

### Cell 1: Check GPU and Install Dependencies

```python
# Check GPU availability
!nvidia-smi

# Clone repository
!git clone https://github.com/roklento/RAG_Chatbot.git
%cd RAG_Chatbot

# IMPORTANT: Install Colab-specific requirements
print("⚙️  Installing dependencies...")
!pip install -q --upgrade "qdrant-client>=1.10.0"
!pip install -q --upgrade "transformers>=4.53.0"
!pip install -q python-dotenv pydantic-settings tiktoken fastembed sentence-transformers accelerate

print("\n✅ Dependencies installed!")
print("\n⚠️  IMPORTANT: Restart runtime now!")
print("   Runtime → Restart runtime")
print("   Then continue from Cell 2")
```

**⚠️ CRITICAL:** After running Cell 1, you MUST restart the runtime before continuing!
- Click: `Runtime → Restart runtime`
- Then start from Cell 2

### Cell 2: Download Models

```python
# Download models from HuggingFace
import os
from huggingface_hub import snapshot_download

# Create models directory
os.makedirs("models", exist_ok=True)

# Download Gemma-3n-E4B-it (for generation + query processing)
print("Downloading Gemma-3n-E4B-it...")
snapshot_download(
    repo_id="google/gemma-3n-E4B-it",
    local_dir="models/gemma-3n-E4B-it",
    local_dir_use_symlinks=False
)

# Download Qwen3-Embedding-4B (for embeddings)
print("Downloading Qwen3-Embedding-4B...")
snapshot_download(
    repo_id="Qwen/Qwen3-Embedding-4B",
    local_dir="models/Qwen3-Embedding-4B",
    local_dir_use_symlinks=False
)

# Download Qwen3-Reranker-4B (for reranking)
print("Downloading Qwen3-Reranker-4B...")
snapshot_download(
    repo_id="Qwen/Qwen3-Reranker-4B",
    local_dir="models/Qwen3-Reranker-4B",
    local_dir_use_symlinks=False
)

print("✅ All models downloaded!")
```

### Cell 3: Configure Environment

```python
# Create .env file for Colab
with open('.env', 'w') as f:
    f.write("""
# Colab Environment Configuration
EMBEDDING_MODEL_PATH=/content/RAG_Chatbot/models/Qwen3-Embedding-4B
RERANKER_MODEL_PATH=/content/RAG_Chatbot/models/Qwen3-Reranker-4B
GENERATION_MODEL_PATH=/content/RAG_Chatbot/models/gemma-3n-E4B-it

DEVICE=cuda
EMBEDDING_DEVICE=cuda
RERANKER_DEVICE=cuda
GENERATION_DEVICE=cuda

QDRANT_HOST=:memory:
QDRANT_PORT=6333

EMBEDDING_DIMENSION=1024
EMBEDDING_BATCH_SIZE=32

ENABLE_QUERY_CORRECTION=true
ENABLE_QUERY_DIVERSIFICATION=true
DIVERSIFICATION_COUNT=2

QA_PAIRS_DENSE_WEIGHT=0.3
QA_PAIRS_SPARSE_WEIGHT=0.7
PLAIN_TEXT_DENSE_WEIGHT=0.7
PLAIN_TEXT_SPARSE_WEIGHT=0.3

ENABLE_RERANKING=true
RERANKER_TOP_K=20

MMR_DIVERSITY_WEIGHT=0.3
MMR_RELEVANCE_WEIGHT=0.7

GENERATION_MODEL_NAME=google/gemma-3n-E4B-it
USE_VLLM=false
USE_TRANSFORMERS=true
GENERATION_DTYPE=bfloat16

MAX_MODEL_LEN=32768
MAX_CONVERSATION_TOKENS=8000
CONVERSATION_WARNING_THRESHOLD=0.8
MAX_CONTEXTS_FOR_GENERATION=5
MAX_RECENT_MESSAGES=10

GENERATION_TEMPERATURE=0.7
GENERATION_TOP_P=0.9
GENERATION_TOP_K=50
GENERATION_MAX_TOKENS=512
GENERATION_MIN_TOKENS=50
ENABLE_STREAMING=true

LOAD_IN_8BIT=false
LOAD_IN_4BIT=false
USE_FLASH_ATTENTION=false
GENERATION_BATCH_SIZE=1

LOG_LEVEL=INFO
VERBOSE=true
""")

print("✅ Environment configured!")
```

### Cell 4: Initialize and Test

```python
import asyncio

# Import and initialize system
from qdrant_client import QdrantClient
from src.config import get_settings
from src.models import QwenEmbedding, QwenReranker, GemmaQueryProcessor
from src.retrieval import create_advanced_retriever
from src.generation import GemmaStreamingGenerator, create_colab_rag_pipeline

print("🚀 Initializing RAG System...")

# Settings
settings = get_settings()

# Qdrant (in-memory)
qdrant_client = QdrantClient(":memory:")
print("✓ Qdrant initialized")

# Embedding model
embedding_model = QwenEmbedding(
    model_path=settings.embedding_model_path,
    device="cuda"
)
print("✓ Embedding model loaded")

# Reranker
reranker = QwenReranker(
    model_path=settings.reranker_model_path,
    device="cuda"
)
print("✓ Reranker loaded")

# Query processor (Gemma)
query_processor = GemmaQueryProcessor(
    model_path=settings.generation_model_path,
    device="cuda"
)
print("✓ Query processor loaded")

print("\n✅ Models loaded successfully!")
```

### Cell 5: Ingest Data

```python
import json
from tqdm import tqdm
from qdrant_client.models import PointStruct, SparseVector, VectorParams, Distance, SparseVectorParams, SparseIndexParams
from fastembed import SparseTextEmbedding
import uuid

print("📥 Ingesting dataset...")

# Setup collections
embedding_dim = 1024

# Create qa_pairs collection
qdrant_client.create_collection(
    collection_name="qa_pairs",
    vectors_config={
        "dense": VectorParams(size=embedding_dim, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False),
        ),
    },
)

# Create plain_text collection
qdrant_client.create_collection(
    collection_name="plain_text",
    vectors_config={
        "dense": VectorParams(size=embedding_dim, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False),
        ),
    },
)

print("✓ Collections created")

# Load data
with open('data/rise_online_qa_pairs.json', 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

with open('data/rise_online_plain_text.json', 'r', encoding='utf-8') as f:
    text_data = json.load(f)

print(f"✓ Loaded {len(qa_data)} Q&A pairs")
print(f"✓ Loaded {len(text_data)} documents")

# Initialize sparse embedding
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# Ingest Q&A pairs
print("\nIngesting Q&A pairs...")
points = []
for qa in tqdm(qa_data):
    text = f"Soru: {qa['question']}\nCevap: {qa['answer']}"

    # Dense embedding
    dense_emb = embedding_model.embed_text(text)

    # Sparse embedding
    sparse_emb = list(sparse_model.embed([text]))[0]
    sparse_vec = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist()
    )

    points.append(PointStruct(
        id=str(uuid.uuid4()),
        vector={"dense": dense_emb, "sparse": sparse_vec},
        payload={
            "question": qa["question"],
            "answer": qa["answer"],
            "combined_text": text,
            "source_type": "qa_pair"
        }
    ))

qdrant_client.upsert(collection_name="qa_pairs", points=points)
print(f"✓ Ingested {len(points)} Q&A pairs")

# Ingest plain text (with chunking)
print("\nIngesting plain text...")
from src.data import create_chunker

chunker = create_chunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk_documents(text_data, "title", "content", "sentence_aware")
print(f"✓ Created {len(chunks)} chunks")

points = []
for chunk in tqdm(chunks):
    # Dense embedding
    dense_emb = embedding_model.embed_text(chunk.content)

    # Sparse embedding
    sparse_emb = list(sparse_model.embed([chunk.content]))[0]
    sparse_vec = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist()
    )

    points.append(PointStruct(
        id=str(uuid.uuid4()),
        vector={"dense": dense_emb, "sparse": sparse_vec},
        payload={
            "content": chunk.content,
            "source_title": chunk.source_title,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "source_type": "plain_text"
        }
    ))

qdrant_client.upsert(collection_name="plain_text", points=points)
print(f"✓ Ingested {len(points)} chunks")

print("\n✅ Data ingestion complete!")
```

### Cell 6: Create RAG Pipeline

```python
# Create retriever
retriever = create_advanced_retriever(
    settings=settings,
    qdrant_client=qdrant_client,
    embedding_model=embedding_model,
    reranker=reranker,
    query_processor=query_processor
)
print("✓ Retriever created")

# Create generator
generator = GemmaStreamingGenerator(
    settings=settings,
    model_path=settings.generation_model_path,
    device="cuda"
)
print("✓ Generator created")

# Create pipeline
pipeline = create_colab_rag_pipeline(
    settings=settings,
    retriever=retriever,
    generator=generator
)

print("\n✅ RAG Pipeline ready!")
```

### Cell 7: Interactive Chat

```python
# Interactive chatbot
session_id = "colab_session"

print("💬 Rise Online RAG Chatbot")
print("-" * 80)
print("Type your questions in Turkish!")
print("Commands: 'quit' to exit, 'reset' to clear history, 'stats' for statistics")
print("-" * 80)

async def chat_loop():
    while True:
        query = input("\n👤 You: ").strip()

        if not query:
            continue

        if query.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break

        if query.lower() == 'reset':
            pipeline.reset_conversation(session_id)
            continue

        if query.lower() == 'stats':
            stats = pipeline.get_conversation_stats(session_id)
            print(f"\n📊 Stats: {stats['message_count']} messages, {stats['total_tokens']:,} tokens ({stats['token_usage_percent']:.1f}%)")
            continue

        # Process query
        print("\n🤖 Assistant: ", end="", flush=True)
        async for result in pipeline.query_stream(query, session_id=session_id, verbose=False):
            if isinstance(result, str):
                print(result, end="", flush=True)
        print()

# Run chat
await chat_loop()
```

---

## 🎬 Demo Queries

Try these sample queries:

```python
demo_queries = [
    "Rise Online nedir?",
    "Hangi karakter sınıfları var?",
    "Savaşçı sınıfı nasıl oynanır?",
    "Seviye nasıl atlanır?",
    "Ekipman nasıl geliştirilir?",
    "PvP sistemi nasıl çalışır?",
]

async def run_demo():
    session_id = "demo"
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*80}")
        print(f"Demo {i}/{len(demo_queries)}: {query}")
        print('='*80)
        print("\n🤖 Assistant: ", end="", flush=True)

        async for result in pipeline.query_stream(query, session_id=session_id, verbose=False):
            if isinstance(result, str):
                print(result, end="", flush=True)
        print("\n")

await run_demo()
```

---

## 💾 Saving and Loading Models (Optional)

### Save to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy models to Drive (one-time)
!cp -r models /content/drive/MyDrive/RAG_Models

print("✅ Models saved to Google Drive!")
```

### Load from Google Drive (next session)

```python
from google.colab import drive
drive.mount('/content/drive')

# Link models from Drive
!ln -s /content/drive/MyDrive/RAG_Models models

print("✅ Models loaded from Google Drive!")
```

---

## 📊 Performance Tips

### Memory Optimization

If running out of VRAM:

```python
# Enable 8-bit quantization
LOAD_IN_8BIT=true

# Or 4-bit quantization (more aggressive)
LOAD_IN_4BIT=true
```

### Speed Optimization

```python
# Reduce context limit
MAX_CONTEXTS_FOR_GENERATION=3  # Instead of 5

# Reduce conversation history
MAX_RECENT_MESSAGES=5  # Instead of 10

# Disable query diversification
ENABLE_QUERY_DIVERSIFICATION=false
```

---

## 🐛 Troubleshooting

### Issue: CUDA out of memory

**Solution:**
```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Restart runtime and use 8-bit quantization
```

### Issue: Model download slow

**Solution:**
```python
# Use Google Drive persistence
# Download once, reuse in future sessions
```

### Issue: Transformers version error

**Solution:**
```python
# Upgrade transformers
!pip install -q --upgrade "transformers>=4.53.0"
```

---

## 📝 Full Notebook Template

A complete Colab notebook (`RAG_Chatbot_Colab.ipynb`) is provided in the repository.

**To use:**
1. Download from repository
2. Upload to Google Colab
3. Run all cells in order
4. Start chatting!

---

## 🎯 Expected Results

After setup:
- **Total time:** ~10-15 minutes (first time)
- **GPU memory:** ~8-12GB (T4 GPU sufficient)
- **Data ingested:** ~75 vectors (30 Q&A + 45 chunks)
- **Response time:** ~2-5 seconds per query
- **Streaming:** Token-by-token display

---

## ✅ Checklist

- [ ] Google Colab with GPU runtime
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Models downloaded
- [ ] Environment configured
- [ ] Data ingested
- [ ] Pipeline initialized
- [ ] Chat working

---

## 🚀 Next Steps

After successful setup:
1. Try demo queries
2. Ask your own questions
3. Test conversation memory
4. Adjust generation parameters
5. Add your own data

---

## 📚 Additional Resources

- **Gemma 3n Documentation:** https://ai.google.dev/gemma/docs/core
- **Qwen Models:** https://github.com/QwenLM/Qwen
- **Project Repository:** https://github.com/YOUR_USERNAME/RAG_Chatbot

---

## 🙏 Support

If you encounter issues:
1. Check GPU is enabled (Runtime → Change runtime type)
2. Verify all models downloaded correctly
3. Check error messages carefully
4. Try restarting runtime

---

**Happy chatting! 🎉**
