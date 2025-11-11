# RAG Chatbot Architecture Documentation

## Overview

This document provides detailed technical documentation of the Advanced RAG Chatbot architecture, focusing on the retrieval pipeline we've built.

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Advanced Retriever                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐ │
│  │   Query    │→ │   Hybrid   │→ │  Reranker  │→ │   MMR    │ │
│  │ Processor  │  │  Retriever │  │            │  │          │ │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Vector Database (Qdrant)                   │
│            ┌──────────────────┐  ┌──────────────────┐          │
│            │   Q&A Pairs      │  │   Plain Text     │          │
│            │   Collection     │  │   Collection     │          │
│            └──────────────────┘  └──────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Query Processor

**Model**: Qwen3-Next-80B-A3B-Instruct

**Responsibilities**:
- Spelling and grammar correction
- Query diversification (generating 3-5 variants)

**Implementation**: `src/models/query_processor.py`

**Process Flow**:
```
Original Query
    ↓
Correction Prompt → LLM → Corrected Query
    ↓
Diversification Prompt → LLM → Query Variants
    ↓
[Corrected Query, Variant 1, Variant 2, Variant 3, ...]
```

**Prompt Templates**:

1. **Correction Prompt**:
```
You are a helpful assistant that corrects spelling and grammar errors.
Task: Correct any spelling or grammar mistakes in the following query.
Rules:
- Fix only obvious errors
- Maintain original intent
- Return ONLY the corrected query
Query: {query}
```

2. **Diversification Prompt**:
```
Task: Generate {N} different versions of the query using different wording.
Rules:
- Same meaning, different phrasing
- Use synonyms and alternative structures
- Make variants diverse
Original query: {query}
```

**Configuration**:
- Temperature: 0.7 (for diversity)
- Max tokens: 512
- Number of variants: 3 (configurable)

---

### 2. Embedding Model

**Model**: Qwen3-Embedding-4B

**Responsibilities**:
- Convert text to dense vectors
- Enable semantic search

**Implementation**: `src/models/embedding.py`

**Features**:
- Embedding dimension: 4096 (full model)
- Matryoshka Representation Learning support
- L2 normalization
- Max sequence length: 8192 tokens
- Multilingual support (100+ languages)

**Usage**:
```python
embedding_model = QwenEmbedding(
    model_path="Qwen/Qwen3-Embedding-4B",
    device="auto",
    max_length=8192,
)

# Single query
query_vec = embedding_model.embed_query("What is ML?")

# Batch documents
doc_vecs = embedding_model.embed_documents([doc1, doc2, doc3])
```

**Technical Details**:
- Uses sentence-transformers library
- Flash Attention 2 support (if available)
- Batch processing for efficiency

---

### 3. Hybrid Retriever

**Implementation**: `src/retrieval/hybrid_retriever.py`

**Dual Collection Strategy**:

| Collection | Dense Weight | Sparse Weight | Rationale |
|------------|--------------|---------------|-----------|
| Q&A Pairs | 30% | 70% | Keyword matching crucial for Q&A |
| Plain Text | 70% | 30% | Semantic understanding more important |

**Multi-Query Retrieval**:
```
For each query variant:
    Search Q&A Collection (top 15)
    Search Text Collection (top 15)
    ↓
Combine all results
    ↓
Apply RRF Fusion
    ↓
Return top 30 candidates
```

**Reciprocal Rank Fusion (RRF)**:
```python
score(doc) = Σ (1 / (k + rank_i))
where:
  k = 60 (constant)
  rank_i = position in ranking i
```

**Why RRF?**
- No score normalization needed
- Handles different scoring scales
- Well-studied in information retrieval
- k=60 is standard from literature

**Implementation**: `src/utils/fusion.py`

---

### 4. Reranker

**Model**: Qwen3-Reranker-4B

**Responsibilities**:
- Cross-encoder relevance scoring
- Precision ranking of candidates

**Implementation**: `src/models/reranker.py`

**Architecture**: Cross-Encoder
- Processes (query, document) pairs jointly
- More accurate than bi-encoders
- Slower but used on small candidate set (30 docs)

**Input Format**:
```
<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}
```

**Process**:
```
30 Candidates
    ↓
Format as (query, doc) pairs
    ↓
Batch process through cross-encoder
    ↓
Extract logits[:, -1, :].max()
    ↓
Apply threshold filter (>0.5)
    ↓
Sort by score
    ↓
Top K results
```

**Features**:
- Custom instructions support (1-5% improvement)
- Batch processing (8 pairs at a time)
- Threshold filtering
- GPU-accelerated

---

### 5. MMR (Maximal Marginal Relevance)

**Implementation**: `src/utils/mmr.py`

**Purpose**: Balance relevance with diversity

**Formula**:
```
MMR = argmax[Di ∈ R \ S] [λ * Sim(Di, Q) - (1-λ) * max[Dj ∈ S] Sim(Di, Dj)]

where:
  Di = candidate document
  Q = query
  S = already selected documents
  λ = relevance weight (0.7 default)
  (1-λ) = diversity weight (0.3 default)
```

**Algorithm**:
```
1. Select most relevant document first
2. For remaining positions:
   - Calculate MMR score for each candidate
   - MMR = relevance - max_similarity_to_selected
   - Select document with highest MMR
3. Repeat until top_k reached
```

**Configuration**:
- Diversity score: 0.3 (30% diversity, 70% relevance)
- Applied after reranking
- Optional (can be disabled)

---

### 6. Qdrant Manager

**Implementation**: `src/retrieval/qdrant_manager.py`

**Responsibilities**:
- Collection creation
- Data ingestion
- Embedding generation
- Batch processing

**Dual Collections**:

1. **Q&A Pairs Collection**:
   - Embeds questions only
   - Stores both question and answer in payload
   - Optimized for exact matching

2. **Plain Text Collection**:
   - Embeds full text
   - Stores full content in payload
   - Optimized for semantic search

**Data Ingestion Pipeline**:
```
Documents
    ↓
Batch (100 docs)
    ↓
Generate Embeddings
    ↓
Create Points (UUID, vector, payload)
    ↓
Upsert to Qdrant
    ↓
Repeat for next batch
```

---

## Complete Retrieval Pipeline

### Step-by-Step Flow

```python
# Input
user_query = "What is masheen lerning?"  # typo intentional

# Step 1: Query Processing
processed = query_processor.process(user_query)
# Output:
# - corrected: "What is machine learning?"
# - variants: [
#     "Explain machine learning",
#     "What does machine learning mean?",
#     "Define machine learning"
#   ]

# Step 2: Multi-Query Hybrid Search
candidates = hybrid_retriever.retrieve(
    queries=[corrected, variant1, variant2, variant3],
    top_k=30
)
# For each query:
#   - Search Q&A (top 15, dense=0.3, sparse=0.7)
#   - Search Text (top 15, dense=0.7, sparse=0.3)
# RRF fuse all results → top 30

# Step 3: Reranking
reranked = reranker.rerank(
    query=corrected,
    documents=[c.content for c in candidates]
)
# Cross-encoder scores each (query, doc) pair
# Filter by threshold (>0.5)

# Step 4: MMR Diversity
final = apply_mmr(
    query_embedding=embed(corrected),
    documents=reranked,
    diversity=0.3,
    top_k=7
)
# Balance relevance and diversity

# Output: Top 7 diverse, relevant results
```

---

## Configuration Parameters

### Model Paths
```
LLM_MODEL_PATH=Qwen/Qwen3-Next-80B-A3B-Instruct
EMBEDDING_MODEL_PATH=Qwen/Qwen3-Embedding-4B
RERANKER_MODEL_PATH=Qwen/Qwen3-Reranker-4B
```

### Retrieval Configuration
```
QUERY_VARIANTS_COUNT=3          # Number of query variants
TOP_K_PER_QUERY=15              # Results per query per collection
CANDIDATES_BEFORE_RERANK=30     # Candidates before reranking
FINAL_TOP_K=7                   # Final results
RERANKER_THRESHOLD=0.5          # Min reranker score
MMR_DIVERSITY_SCORE=0.3         # Diversity weight
```

### Hybrid Search Weights
```
QA_DENSE_WEIGHT=0.3             # Q&A: Dense
QA_SPARSE_WEIGHT=0.7            # Q&A: Sparse
TEXT_DENSE_WEIGHT=0.7           # Text: Dense
TEXT_SPARSE_WEIGHT=0.3          # Text: Sparse
```

### Why These Values?

**Query Variants (3)**:
- Balance between coverage and latency
- More variants = better recall but slower

**Candidates Before Rerank (30)**:
- Enough for diversity
- Small enough for fast reranking
- Sweet spot for cross-encoder efficiency

**Final Top K (7)**:
- Enough context for LLM
- Not overwhelming
- Fits in most prompts

**Reranker Threshold (0.5)**:
- Filters low-quality matches
- Based on empirical testing
- Can be adjusted per use case

**Hybrid Weights**:
- Q&A sparse-heavy: Users often use similar wording
- Text dense-heavy: Semantic understanding crucial
- Validated through testing

---

## Performance Characteristics

### Latency (Estimated)

| Stage | Time | Notes |
|-------|------|-------|
| Query Processing | 2-4s | 80B LLM, generates 3 variants |
| Embedding (1 query) | <0.1s | 4B model, cached after first use |
| Qdrant Search (3 queries × 2 collections) | 0.2-0.5s | Depends on DB size |
| RRF Fusion | <0.01s | Pure Python, fast |
| Reranking (30 docs) | 0.3-0.6s | 4B cross-encoder, batch=8 |
| MMR | <0.1s | Numpy operations |
| **Total** | **2.5-5s** | Per query end-to-end |

### Optimization Opportunities

1. **Query Processing** (biggest bottleneck):
   - Use smaller model (Qwen2.5-14B): ~5x faster
   - Cache common corrections
   - Async processing

2. **Embedding**:
   - Batch multiple user queries
   - Cache query embeddings

3. **Reranking**:
   - Reduce candidates (20 instead of 30)
   - Use smaller reranker for speed

---

## Extensibility

### Adding New Components

1. **Custom Query Processor**:
```python
class CustomQueryProcessor(QueryProcessor):
    def correct_query(self, query: str) -> str:
        # Your custom logic
        return corrected_query
```

2. **Custom Retriever**:
```python
class CustomRetriever(HybridRetriever):
    def _hybrid_search_collection(self, ...):
        # Your custom search logic
        return results
```

3. **Custom Reranker**:
```python
class CustomReranker(QwenReranker):
    def rerank(self, query, documents):
        # Your custom reranking logic
        return ranked_docs
```

### Future Enhancements

1. **True Sparse Vector Support**:
   - Configure Qdrant with sparse vectors
   - Use BM25 or SPLADE embeddings
   - Server-side hybrid fusion

2. **Caching Layer**:
   - Redis for query caching
   - Embedding cache
   - Result cache for common queries

3. **Async Processing**:
   - Concurrent query variant processing
   - Parallel collection search
   - Async reranking

4. **Monitoring**:
   - Latency tracking per component
   - Quality metrics
   - A/B testing infrastructure

---

## Best Practices

### 1. Data Preparation
- Clean text before ingestion
- Meaningful metadata
- Appropriate chunking for long documents

### 2. Configuration Tuning
- Start with defaults
- Monitor retrieval quality
- Adjust weights based on your data

### 3. Model Selection
- Consider latency requirements
- GPU memory constraints
- Quality vs speed tradeoffs

### 4. Production Deployment
- Use vLLM/SGLang for LLM inference
- Load balance across multiple instances
- Implement request queuing

---

## Troubleshooting

### Common Issues

1. **Slow Query Processing**:
   - Switch to smaller LLM
   - Reduce query variants
   - Enable caching

2. **Poor Retrieval Quality**:
   - Check collection weights
   - Verify data quality
   - Adjust reranker threshold

3. **Out of Memory**:
   - Use model quantization
   - Reduce batch sizes
   - Use CPU for smaller models

4. **Qdrant Connection Issues**:
   - Verify Qdrant is running
   - Check host/port settings
   - Review firewall settings

---

## References

- [Qwen3-Next Models](https://huggingface.co/Qwen)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Maximal Marginal Relevance](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)

---

*This architecture documentation covers the retrieval component. The generation component will be added in the next phase.*
