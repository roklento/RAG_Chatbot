# Data Ingestion Guide

Complete guide for creating, processing, and ingesting data into the RAG chatbot system.

---

## Overview

The data ingestion pipeline consists of:

1. **Dataset Creation** - Q&A pairs and plain text documents
2. **Text Chunking** - Split long texts into manageable pieces
3. **Embedding Generation** - Create dense (semantic) and sparse (BM25) vectors
4. **Qdrant Ingestion** - Store vectors in dual collections

---

## Dataset Format

### Q&A Pairs (`data/rise_online_qa_pairs.json`)

Question-answer pairs for exact matching and factual queries:

```json
[
  {
    "question": "Rise Online nedir?",
    "answer": "Rise Online, Türk yapımı bir MMORPG oyunudur..."
  },
  {
    "question": "Kaç karakter sınıfı var?",
    "answer": "Rise Online'da 3 ana karakter sınıfı bulunmaktadır..."
  }
]
```

**Characteristics:**
- Short, focused Q&A format
- Direct answers to specific questions
- 70% sparse / 30% dense weighting (keyword-focused)
- Total: 30 pairs in dataset

### Plain Text Documents (`data/rise_online_plain_text.json`)

Long-form articles and guides for semantic search:

```json
[
  {
    "title": "Rise Online - Genel Bakış",
    "content": "Rise Online, Türkiye'de geliştirilmiş... [long text]"
  },
  {
    "title": "Karakter Sınıfları - Detaylı Rehber",
    "content": "Rise Online'da üç ana karakter sınıfı... [long text]"
  }
]
```

**Characteristics:**
- Long-form content (500-2000+ characters)
- Detailed explanations and guides
- Gets chunked into 512-character pieces
- 30% sparse / 70% dense weighting (semantic-focused)
- Total: 8 documents in dataset

---

## Text Chunking

Plain text documents are automatically chunked for optimal retrieval.

### Chunking Strategy

**Sentence-Aware Chunking** (default):
- Respects sentence boundaries (doesn't break mid-sentence)
- Target chunk size: 512 characters
- Overlap: 50 characters between chunks
- Minimum chunk size: 100 characters

**Example:**
```
Original document (1500 chars)
    ↓
Chunk 1 (512 chars) ───┐
Chunk 2 (512 chars)    ├── 50 char overlap
Chunk 3 (476 chars) ───┘
```

### Chunking Module

Located at `src/data/text_chunker.py`:

```python
from src.data import create_chunker

chunker = create_chunker(
    chunk_size=512,
    chunk_overlap=50,
    min_chunk_size=100,
)

chunks = chunker.chunk_documents(
    documents=text_data,
    strategy="sentence_aware",
)
```

**Strategies Available:**
- `sentence_aware` - Respects sentence boundaries (recommended)
- `paragraph` - Chunks by paragraphs
- `character` - Simple character-based splitting

---

## Embedding Generation

### Dense Embeddings (Semantic Vectors)

**Model:** Qwen3-Embedding-4B
- Dimension: 1024
- Type: Dense float vectors
- Distance: Cosine similarity
- Purpose: Semantic similarity search

```python
from src.models import QwenEmbedding

embedding_model = QwenEmbedding(
    model_path="path/to/Qwen3-Embedding-4B",
    device="cuda",
)

vector = embedding_model.embed_text("Your text here")
# Returns: List[float] of length 1024
```

### Sparse Embeddings (BM25 Vectors)

**Model:** Qdrant/bm25 (via fastembed)
- Type: Sparse vectors (indices + values)
- Purpose: Keyword matching and exact term search

```python
from fastembed import SparseTextEmbedding

sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
sparse_vector = list(sparse_model.embed(["Your text here"]))[0]
# Returns: SparseVector with indices and values
```

---

## Qdrant Collections

### Collection Structure

Two collections with different optimization profiles:

| Collection | Dense % | Sparse % | Purpose |
|-----------|---------|----------|---------|
| `qa_pairs` | 30% | 70% | Keyword-focused, exact matches |
| `plain_text` | 70% | 30% | Semantic search, understanding |

### Collection Schema

**qa_pairs:**
```python
{
    "id": "uuid",
    "vector": {
        "dense": [1024 floats],
        "sparse": {"indices": [...], "values": [...]}
    },
    "payload": {
        "question": "Soru metni",
        "answer": "Cevap metni",
        "combined_text": "Soru + Cevap",
        "source_type": "qa_pair",
        "index": 0
    }
}
```

**plain_text:**
```python
{
    "id": "uuid",
    "vector": {
        "dense": [1024 floats],
        "sparse": {"indices": [...], "values": [...]}
    },
    "payload": {
        "content": "Chunk içeriği",
        "source_title": "Belge başlığı",
        "chunk_index": 0,
        "total_chunks": 5,
        "source_type": "plain_text",
        "metadata": {}
    }
}
```

---

## Step-by-Step Ingestion

### Prerequisites

1. **Qdrant Running:**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Environment Variables:**
   ```bash
   # Copy and configure
   cp .env.example .env

   # Essential settings:
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   EMBEDDING_MODEL_PATH=/path/to/Qwen3-Embedding-4B
   DEVICE=cuda  # or cpu
   ```

3. **Dependencies Installed:**
   ```bash
   pip install -r requirements.txt
   ```

### Step 1: Setup Collections

Create Qdrant collections with proper configuration:

```bash
cd scripts
python setup_qdrant_collections.py
```

**What it does:**
- Connects to Qdrant
- Creates `qa_pairs` collection (1024-dim dense + sparse)
- Creates `plain_text` collection (1024-dim dense + sparse)
- Configures indexes and parameters

**Output:**
```
QDRANT COLLECTION SETUP
Connecting to Qdrant at localhost:6333...
✓ Connected

Setting up collection: qa_pairs
  ✓ Created collection
  - Dense vector dimension: 1024
  - Sparse vector support: enabled

Setting up collection: plain_text
  ✓ Created collection
  - Dense vector dimension: 1024
  - Sparse vector support: enabled

SETUP COMPLETE
```

### Step 2: Ingest Data

Load, chunk, embed, and insert all data:

```bash
python ingest_data.py
```

**What it does:**
1. Loads JSON datasets from `data/` directory
2. For Q&A pairs:
   - Combines question + answer
   - Generates dense embeddings
   - Generates sparse vectors
   - Inserts into `qa_pairs` collection
3. For plain text:
   - Chunks documents (sentence-aware)
   - Generates embeddings for each chunk
   - Generates sparse vectors
   - Inserts into `plain_text` collection

**Progress Output:**
```
DATA INGESTION PIPELINE

Loading embedding model...
✓ Model loaded (1024 dimensions)

Loading datasets...
  ✓ Loaded 30 Q&A pairs
  ✓ Loaded 8 plain text documents

INGESTING Q&A PAIRS
  Generating dense embeddings...
  Dense embedding: 100%|████████| 30/30
  Generating sparse vectors (BM25)...
  Sparse embedding: 100%|████████| 30/30
  ✓ Inserted 30 Q&A pairs

INGESTING PLAIN TEXT DOCUMENTS
  Chunking documents...
  ✓ Created 45 chunks

  Chunk Statistics:
    Total chunks: 45
    Avg length: 487 characters
    Min length: 312 characters
    Max length: 512 characters
    Unique sources: 8

  Generating dense embeddings...
  Dense embedding: 100%|████████| 45/45
  Generating sparse vectors (BM25)...
  Sparse embedding: 100%|████████| 45/45
  ✓ Inserted 45 chunks

INGESTION COMPLETE

Collection Summary:
  qa_pairs: 30 points
  plain_text: 45 points

✓ Data ingestion successful!
```

### Step 3: Verify Ingestion

Check that data was ingested correctly:

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Check collections
qa_info = client.get_collection("qa_pairs")
print(f"Q&A pairs: {qa_info.points_count} points")

text_info = client.get_collection("plain_text")
print(f"Plain text: {text_info.points_count} points")

# Sample a point
points = client.scroll(
    collection_name="qa_pairs",
    limit=1,
)[0]

print(f"\nSample Q&A pair:")
print(f"Question: {points[0].payload['question']}")
print(f"Answer: {points[0].payload['answer'][:100]}...")
```

---

## Adding Your Own Data

### Creating Q&A Pairs

1. Create JSON file in `data/` directory:

```json
[
  {
    "question": "Your question in Turkish?",
    "answer": "Your detailed answer in Turkish..."
  }
]
```

2. Add to ingestion script or create separate script:

```python
new_qa_data = load_json_data("data/your_qa_pairs.json")

ingest_qa_pairs(
    client=client,
    embedding_model=embedding_model,
    qa_data=new_qa_data,
)
```

### Creating Plain Text Documents

1. Create JSON file:

```json
[
  {
    "title": "Document Title",
    "content": "Long form content that will be automatically chunked..."
  }
]
```

2. Ingest with automatic chunking:

```python
new_text_data = load_json_data("data/your_documents.json")

ingest_plain_text(
    client=client,
    embedding_model=embedding_model,
    text_data=new_text_data,
    chunk_size=512,  # Adjust as needed
    chunk_overlap=50,
)
```

---

## Dataset Statistics

### Current Dataset

**Q&A Pairs:**
- Total: 30 pairs
- Average question length: ~35 characters
- Average answer length: ~280 characters
- Topics: Game basics, classes, leveling, equipment, PvP, PvE, economy, guilds

**Plain Text:**
- Total: 8 documents
- Total characters: ~12,000
- After chunking: ~45 chunks
- Average chunk size: ~487 characters
- Topics: Game overview, class guides, leveling strategies, equipment systems, dungeons, PvP, economy, guilds, advanced mechanics

---

## Troubleshooting

### Issue: "Collection already exists"

**Solution:**
```bash
# Delete and recreate
python -c "from qdrant_client import QdrantClient; c = QdrantClient(host='localhost'); c.delete_collection('qa_pairs'); c.delete_collection('plain_text')"

# Then run setup again
python scripts/setup_qdrant_collections.py
```

### Issue: "Embedding model not found"

**Solution:**
1. Download Qwen3-Embedding-4B model
2. Update `.env` with correct path:
   ```
   EMBEDDING_MODEL_PATH=/absolute/path/to/Qwen3-Embedding-4B
   ```

### Issue: "Out of memory during embedding"

**Solution:**
1. Use CPU instead of GPU:
   ```
   DEVICE=cpu
   ```
2. Process in smaller batches (modify `ingest_data.py`)

### Issue: "Qdrant connection failed"

**Solution:**
1. Check if Qdrant is running:
   ```bash
   docker ps | grep qdrant
   ```
2. Start Qdrant:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

---

## Next Steps

After successful ingestion:

1. **Test Retrieval:**
   ```bash
   cd examples
   python 03_test_retrieval.py
   ```

2. **Start Full RAG Chatbot:**
   ```bash
   # Make sure vLLM server is running
   bash examples/04_setup_vllm.sh

   # Start chatbot
   python examples/08_complete_rag_demo.py --interactive
   ```

3. **Add More Data:**
   - Create more Q&A pairs for specific topics
   - Add detailed guides and documentation
   - Re-run ingestion script

---

## Files Created

### Dataset Files:
- `data/rise_online_qa_pairs.json` - 30 Q&A pairs
- `data/rise_online_plain_text.json` - 8 documents

### Code Modules:
- `src/data/text_chunker.py` - Text chunking module
- `src/data/__init__.py` - Data module exports

### Scripts:
- `scripts/setup_qdrant_collections.py` - Collection setup
- `scripts/ingest_data.py` - Complete ingestion pipeline

### Documentation:
- `DATA_INGESTION.md` - This file

---

## Performance Notes

### Ingestion Speed

On typical hardware:
- **Q&A pairs (30):** ~30-60 seconds
- **Plain text (45 chunks):** ~60-90 seconds
- **Total:** ~2-3 minutes for full dataset

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB (for models)

**Recommended:**
- GPU: NVIDIA with 8GB+ VRAM
- RAM: 16GB+
- Storage: 20GB+ SSD

### Scaling

For larger datasets (1000+ documents):
- Use batch processing
- Consider distributed embedding generation
- Use Qdrant's collection sharding
- Monitor memory usage

---

## Summary

✅ **Completed:**
- [x] Synthetic Rise Online dataset (Turkish)
- [x] Q&A pairs (30) and plain text documents (8)
- [x] Text chunking module with sentence-aware strategy
- [x] Dual embedding (dense + sparse)
- [x] Qdrant dual collections (optimized weights)
- [x] Complete ingestion pipeline
- [x] Comprehensive documentation

✅ **Ready for:**
- Retrieval testing
- Full RAG chatbot usage
- Adding custom data
- Production deployment
