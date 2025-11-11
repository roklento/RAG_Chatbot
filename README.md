# Advanced RAG Chatbot

An advanced Retrieval-Augmented Generation (RAG) chatbot system with state-of-the-art retrieval techniques.

## 🌟 Features

### Advanced Retrieval Pipeline

1. **Query Processing**
   - Automatic spelling and grammar correction
   - Query diversification (3-5 semantic variants)
   - Context-aware query understanding
   - Powered by Qwen3-Next-80B-A3B-Instruct

2. **Hybrid Search**
   - Dense (semantic) + Sparse (keyword) retrieval
   - Dual collection architecture (Q&A pairs vs Plain text)
   - Collection-specific weight optimization
   - Multi-query fusion with Reciprocal Rank Fusion (RRF)

3. **Reranking**
   - Cross-encoder reranking with Qwen3-Reranker-4B
   - Relevance threshold filtering
   - Instruction-based tuning support

4. **Diversity Optimization**
   - Maximal Marginal Relevance (MMR) filtering
   - Balances relevance with diversity
   - Configurable diversity weights

## 🏗️ Architecture

```
User Query
    ↓
┌─────────────────────────────────────┐
│ Query Processing (Qwen3-Next-80B)   │
│  - Correction                        │
│  - 3-5 variants generation           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Hybrid Multi-Query Search           │
│  ┌──────────────┐  ┌──────────────┐│
│  │ Q&A Pairs    │  │ Plain Text   ││
│  │ Dense: 30%   │  │ Dense: 70%   ││
│  │ Sparse: 70%  │  │ Sparse: 30%  ││
│  └──────────────┘  └──────────────┘│
│         RRF Fusion                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Reranking (Qwen3-Reranker-4B)       │
│  - Cross-encoder scoring             │
│  - Threshold filtering (>0.5)        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ MMR Diversity Filtering             │
│  - Balance relevance & diversity     │
└─────────────────────────────────────┘
    ↓
Final Results (Top 5-10)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for 80B model)
- Qdrant vector database

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG_Chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Setup Qdrant:
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or install locally
# See: https://qdrant.tech/documentation/guides/installation/
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

### Usage

#### 1. Setup Database

```bash
cd examples
python 01_setup_database.py
```

This will:
- Create Qdrant collections
- Add sample Q&A pairs
- Add sample plain text documents

#### 2. Test Retrieval

```bash
python 02_test_retrieval.py
```

This demonstrates the full retrieval pipeline with verbose output.

#### 3. Use in Your Code

```python
from src.config import get_settings
from src.retrieval.advanced_retriever import create_advanced_retriever

# Initialize
settings = get_settings()
retriever = create_advanced_retriever(settings)

# Simple retrieval
results = retriever.retrieve_simple(
    query="What is machine learning?",
    top_k=5
)

for doc in results:
    print(doc)

# Detailed retrieval
results = retriever.retrieve(
    query="Explain transformers",
    top_k=5,
    apply_mmr=True,
    verbose=True
)

for result in results:
    print(f"Score: {result.relevance_score:.4f}")
    print(f"Content: {result.content}")
    print(f"Collection: {result.collection}")
```

## 📁 Project Structure

```
RAG_Chatbot/
├── src/
│   ├── config/           # Configuration management
│   ├── models/           # Model wrappers
│   │   ├── embedding.py      # Qwen3-Embedding-4B
│   │   ├── reranker.py       # Qwen3-Reranker-4B
│   │   └── query_processor.py # Qwen3-Next-80B
│   ├── retrieval/        # Retrieval components
│   │   ├── qdrant_manager.py      # Database management
│   │   ├── hybrid_retriever.py    # Hybrid search
│   │   └── advanced_retriever.py  # Main orchestrator
│   └── utils/            # Utility functions
│       ├── fusion.py     # RRF implementation
│       └── mmr.py        # MMR implementation
├── examples/             # Example scripts
├── tests/               # Test files
├── data/                # Data directory
├── requirements.txt     # Dependencies
├── .env.example        # Environment template
└── README.md           # This file
```

## ⚙️ Configuration

Key configuration parameters in `.env`:

```bash
# Models
LLM_MODEL_PATH=Qwen/Qwen3-Next-80B-A3B-Instruct
EMBEDDING_MODEL_PATH=Qwen/Qwen3-Embedding-4B
RERANKER_MODEL_PATH=Qwen/Qwen3-Reranker-4B

# Retrieval Settings
QUERY_VARIANTS_COUNT=3          # Number of query variants
TOP_K_PER_QUERY=15              # Results per query per collection
CANDIDATES_BEFORE_RERANK=30     # Candidates before reranking
FINAL_TOP_K=7                   # Final results after all filtering
RERANKER_THRESHOLD=0.5          # Min reranker score
MMR_DIVERSITY_SCORE=0.3         # Diversity weight (0-1)

# Hybrid Search Weights
QA_DENSE_WEIGHT=0.3             # Q&A: Dense search weight
QA_SPARSE_WEIGHT=0.7            # Q&A: Sparse search weight
TEXT_DENSE_WEIGHT=0.7           # Text: Dense search weight
TEXT_SPARSE_WEIGHT=0.3          # Text: Sparse search weight

# RRF
RRF_K=60                        # RRF constant (standard: 60)
```

## 🔬 Models Used

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Query Processing | Qwen3-Next-80B-A3B-Instruct | 80B | Correction & diversification |
| Embedding | Qwen3-Embedding-4B | 4B | Semantic embeddings |
| Reranking | Qwen3-Reranker-4B | 4B | Relevance scoring |

## 📊 Performance Considerations

### Latency Breakdown (Estimated)

- **Query Processing**: 2-4s (80B LLM)
- **Hybrid Search**: 0.2-0.5s (Qdrant)
- **Reranking**: 0.3-0.6s (4B model, 30 candidates)
- **MMR**: <0.1s (numpy operations)
- **Total**: ~3-5 seconds per query

### Optimization Tips

1. **For Production**: Consider using smaller model (Qwen2.5-14B) for query processing
2. **GPU Memory**: 80B model requires ~80GB VRAM (quantized) or ~160GB (FP16)
3. **Batch Processing**: Process multiple queries in parallel when possible
4. **Caching**: Cache query embeddings for repeated queries

## 🧪 Testing

Run component tests:

```bash
cd examples
python 03_component_testing.py
```

This allows testing individual components:
- Query processor
- Embedding model
- Reranker
- Hybrid retriever

## 📚 Advanced Usage

### Custom Data Ingestion

```python
from src.config import get_settings
from src.models import QwenEmbedding
from src.retrieval import QdrantManager

settings = get_settings()
embedding_model = QwenEmbedding(settings.embedding_model_path)
manager = QdrantManager(settings, embedding_model)

# Add Q&A pairs
qa_pairs = [
    {"question": "...", "answer": "..."},
    # ...
]
manager.add_qa_pairs(qa_pairs)

# Add plain text
texts = ["Document 1...", "Document 2...", ...]
manager.add_plain_text(texts)
```

### Custom Retrieval Pipeline

```python
from src.retrieval import AdvancedRetriever

# Custom settings
results = retriever.retrieve(
    query="Your query",
    top_k=10,              # More results
    apply_mmr=False,       # Disable diversity
    verbose=True,          # Show pipeline steps
)
```

## 🤝 Contributing

This is a demonstration project showcasing advanced RAG techniques. Feel free to extend and customize for your use case.

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- Qwen Team for excellent open-source models
- Qdrant for the vector database
- LangChain and LlamaIndex communities

## 📞 Support

For issues and questions, please open an issue on the repository.

---

**Note**: This is the retrieval component of the RAG system. The generation component (using the same Qwen3-Next-80B model) will be implemented in the next phase.
