# Generation System Documentation (Phase 2A)

## Overview

The generation system implements a complete RAG pipeline with streaming support, conversation memory, and advanced response processing. Built on top of the retrieval system from Phase 1.

---

## Architecture Components

### 1. **Conversation Memory Manager** (`src/generation/conversation_memory.py`)

Manages conversation history with token-aware tracking.

**Features:**
- Per-session conversation history
- Accurate token counting with tiktoken (cl100k_base encoding)
- Automatic truncation when approaching limits
- 200K token safe limit (256K max with buffer)
- Keep last 20 messages at full detail

**Usage:**
```python
from src.generation import ConversationMemoryManager

memory = ConversationMemoryManager(settings)

# Add messages
memory.add_message("session_1", "user", "Merhaba")
memory.add_message("session_1", "assistant", "Merhaba! Nasıl yardımcı olabilirim?")

# Get history
history = memory.get_history("session_1")

# Check tokens
total_tokens = memory.get_total_tokens("session_1")

# Check if should reset
if memory.should_reset("session_1"):
    memory.reset_session("session_1")
```

---

### 2. **Context Augmenter** (`src/generation/context_augmenter.py`)

Prepares retrieved contexts for generation with citation tracking.

**Features:**
- Assigns citation IDs [1], [2], [3]...
- Tracks token count per context
- Limits to max contexts (default: 7)
- Formats contexts for prompt

**Usage:**
```python
from src.generation import ContextAugmenter

augmenter = ContextAugmenter(settings)

# Augment retrieved results
augmented = augmenter.augment_contexts(
    retrieved_results,
    max_contexts=7
)

# Format for prompt
formatted_text = augmenter.format_for_prompt(augmented)
```

---

### 3. **Prompt Builder** (`src/generation/prompt_builder.py`)

Builds comprehensive prompts with Turkish system instructions.

**Features:**
- Advanced Turkish system prompt adapted from RiseBot
- Combines system prompt + contexts + history + query
- Clear citation rules
- Pure text response formatting

**System Prompt Rules:**
- Default language: Turkish
- Use ONLY provided contexts for factual questions
- Cite sources with [1], [2], [3]
- Pure text responses (no markdown, code blocks, HTML)
- Consider conversation history

**Usage:**
```python
from src.generation import PromptBuilder

builder = PromptBuilder(settings)

prompt = builder.build_prompt(
    query="Makine öğrenmesi nedir?",
    augmented_contexts=augmented_contexts,
    conversation_history=history
)
```

---

### 4. **vLLM Streaming Generator** (`src/generation/vllm_generator.py`)

Async streaming generator using vLLM for efficient inference.

**Features:**
- Token-by-token async streaming
- OpenAI-compatible API
- Multi-token prediction support (2x faster)
- Efficient MoE inference (3B active params from 80B total)
- Health checking
- Fallback to transformers if vLLM unavailable

**Usage:**
```python
from src.generation import VLLMStreamingGenerator, GenerationConfig

generator = VLLMStreamingGenerator(settings)

# Check health
is_healthy = await generator.health_check()

# Stream generation
config = GenerationConfig(
    temperature=0.7,
    top_p=0.9,
    max_tokens=1024,
)

async for token in generator.generate_stream(prompt, config):
    print(token, end="", flush=True)

# Or generate complete response
response = await generator.generate(prompt, config)
```

---

### 5. **Response Post-Processor** (`src/generation/post_processor.py`)

Validates and processes generated responses.

**Features:**
- Extract and validate citations
- Remove markdown/HTML formatting
- Calculate confidence scores
- Track source usage
- Generate processing metrics

**Confidence Scoring Factors:**
- Citation usage ratio (0-0.5 points)
- Absolute citation count (0-0.3 points)
- Response length (0-0.1 points)
- Uncertainty detection (-0.2 points)
- No citations penalty (-0.3 points)

**Usage:**
```python
from src.generation import ResponsePostProcessor

processor = ResponsePostProcessor(settings)

processed = processor.process(
    generated_text=generated_text,
    available_citations=[1, 2, 3],
    context_details=context_details
)

print(f"Confidence: {processed.confidence_score:.2f}")
print(f"Citations used: {processed.citations_used}")
print(f"Sources coverage: {processed.sources_coverage:.1%}")
```

---

### 6. **Streaming RAG Pipeline** (`src/generation/rag_pipeline.py`)

Complete end-to-end RAG orchestration.

**Pipeline Flow:**
1. Check conversation history and token limits
2. Retrieve relevant contexts (using AdvancedRetriever)
3. Augment contexts with citations
4. Build comprehensive prompt
5. Generate streaming response
6. Post-process and validate
7. Update conversation history

**Features:**
- Full pipeline orchestration
- Streaming and non-streaming modes
- Conversation management
- Health checking
- Performance metrics

**Usage:**
```python
from src.generation import create_rag_pipeline

pipeline = create_rag_pipeline(
    settings=settings,
    retriever=retriever,
    generator=generator
)

# Streaming mode
async for result in pipeline.query_stream(query, session_id="user_1"):
    if isinstance(result, str):
        print(result, end="", flush=True)  # Token
    else:
        final_response = result  # RAGResponse with metadata

# Non-streaming mode
response = await pipeline.query(query, session_id="user_1")

# Get conversation stats
stats = pipeline.get_conversation_stats("user_1")

# Health check
health = await pipeline.health_check()
```

---

## Configuration

All generation settings in `.env`:

```bash
# vLLM Server
VLLM_SERVER_URL=http://localhost:8000
VLLM_MODEL_NAME=Qwen/Qwen3-Next-80B-A3B-Instruct
VLLM_MAX_MODEL_LEN=262144  # 256K context

# Generation
GENERATION_TEMPERATURE=0.7
GENERATION_TOP_P=0.9
GENERATION_MAX_TOKENS=1024
GENERATION_MIN_TOKENS=50
ENABLE_STREAMING=true

# Conversation History
MAX_CONVERSATION_TOKENS=200000  # 200K safe limit
CONVERSATION_WARNING_THRESHOLD=0.8
MAX_RECENT_MESSAGES=20
MAX_CONTEXTS_FOR_GENERATION=7
```

---

## Setting Up vLLM Server

### Option 1: Using Setup Script (Recommended)

```bash
# Start vLLM server
bash examples/04_setup_vllm.sh
```

### Option 2: Manual Setup

```bash
# Install vLLM
pip install vllm>=0.6.0

# Run server
vllm serve /path/to/Qwen3-Next-80B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 262144 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
```

### Configuration Options

- `--tensor-parallel-size`: Number of GPUs (adjust based on hardware)
- `--max-model-len`: Maximum context length (256K for Qwen3-Next)
- `--gpu-memory-utilization`: GPU memory usage (0.90 = 90%)
- `--speculative-config`: Enable multi-token prediction for 2x faster streaming

---

## Example Scripts

### 1. Test Generation Only (`examples/05_test_generation.py`)

Tests generator component without retrieval:
- Health checks
- Streaming generation
- Non-streaming generation
- Custom configurations

```bash
python examples/05_test_generation.py
```

### 2. Test Streaming Components (`examples/06_test_streaming.py`)

Tests individual streaming components:
- Conversation memory
- Context augmentation
- Prompt building
- Post-processing

```bash
python examples/06_test_streaming.py
```

### 3. Conversation Demo (`examples/07_conversation_demo.py`)

Multi-turn conversation without retrieval:
- Scripted conversation demo
- Interactive mode
- Token tracking

```bash
# Scripted demo
python examples/07_conversation_demo.py

# Interactive mode
python examples/07_conversation_demo.py --interactive
```

### 4. Complete RAG Demo (`examples/08_complete_rag_demo.py`)

**Full end-to-end RAG chatbot** with all features:
- Query processing + diversification
- Hybrid retrieval (dual collections)
- Reranking + MMR
- Context augmentation
- Conversation history
- Streaming generation
- Response post-processing

```bash
# Run all demos
python examples/08_complete_rag_demo.py

# Interactive chatbot
python examples/08_complete_rag_demo.py --interactive

# Health check only
python examples/08_complete_rag_demo.py --health
```

---

## Complete RAG Flow

```
User Query
    ↓
[Query Processor] → Correction + Diversification
    ↓
[Hybrid Retriever] → Dense + Sparse Search (2 collections)
    ↓
[Reranker] → Relevance scoring
    ↓
[MMR Filter] → Diversity filtering
    ↓
[Context Augmenter] → Add citations [1], [2], [3]
    ↓
[Conversation Memory] → Get history + check tokens
    ↓
[Prompt Builder] → System prompt + contexts + history + query
    ↓
[vLLM Generator] → Stream tokens
    ↓
[Post-Processor] → Validate citations + format + score
    ↓
[Update Memory] → Save to conversation history
    ↓
Final Response
```

---

## Performance Metrics

### RAGResponse Metadata

Every query returns comprehensive metadata:

```python
response = await pipeline.query(query)

# Processed response
response.processed_response.text  # Final text
response.processed_response.confidence_score  # 0-1
response.processed_response.citations_used  # [1, 2, 3]
response.processed_response.sources_coverage  # 0-1

# Retrieved contexts
response.retrieved_contexts  # Original retrieval results
response.augmented_contexts  # With citations

# Performance
response.retrieval_time_ms  # Retrieval latency
response.generation_time_ms  # Generation latency
response.total_time_ms  # Total pipeline time

# Conversation
response.conversation_tokens  # Total tokens in history
```

---

## Key Features

### ✅ **Conversation Memory**
- 256K context window (200K safe limit)
- Accurate token counting with tiktoken
- Automatic truncation and warnings
- Per-session management

### ✅ **Streaming Support**
- Token-by-token async streaming
- Real-time response display
- Low latency with vLLM
- Multi-token prediction (2x faster)

### ✅ **Citation Tracking**
- Automatic [1], [2], [3] citations
- Validation against provided contexts
- Source coverage metrics
- Citation detail tracking

### ✅ **Response Quality**
- Confidence scoring
- Formatting validation (remove markdown/HTML)
- Uncertainty detection
- Processing notes

### ✅ **Turkish Language**
- Advanced Turkish system prompt
- Natural conversation flow
- Context-aware responses
- Bilingual support (Turkish primary, English fallback)

---

## Model Information

### Qwen3-Next-80B-A3B-Instruct

- **Type:** Mixture of Experts (MoE)
- **Total Parameters:** 80B
- **Active Parameters:** 3B per token
- **Context Window:** 256K tokens (extendable to 1M)
- **Performance:** State-of-the-art on Turkish benchmarks
- **Inference:** vLLM for efficiency

**Why MoE?**
- Only 3B params active = fast inference
- 80B total = strong capabilities
- Best of both worlds: speed + quality

---

## Next Steps (Phase 2B)

Potential enhancements:
- [ ] Add response caching
- [ ] Implement query suggestions
- [ ] Add user feedback collection
- [ ] Create web interface
- [ ] Add response regeneration
- [ ] Implement conversation branching
- [ ] Add response scoring UI
- [ ] Create API endpoints

---

## Troubleshooting

### vLLM Server Not Available

```bash
# Check if server is running
curl http://localhost:8000/health

# Check vLLM logs
# Start server with: bash examples/04_setup_vllm.sh
```

### Token Limit Exceeded

```python
# Check conversation stats
stats = pipeline.get_conversation_stats(session_id)
print(f"Usage: {stats['token_usage_percent']:.1f}%")

# Reset if needed
pipeline.reset_conversation(session_id)
```

### Low Confidence Scores

- Check if contexts are relevant
- Verify citations are being used
- Review system prompt adherence
- Check response length

---

## Implementation Status

**Phase 2A: ✅ COMPLETED**

- ✅ Conversation memory with token tracking
- ✅ Context augmentation with citations
- ✅ Turkish prompt builder
- ✅ vLLM streaming generator
- ✅ Response post-processor
- ✅ Complete RAG pipeline
- ✅ Example scripts and demos
- ✅ Documentation

**Ready for:** Production use or Phase 2B enhancements

---

## Files Created

### Core Components (6 files)
1. `src/generation/conversation_memory.py`
2. `src/generation/context_augmenter.py`
3. `src/generation/prompt_builder.py`
4. `src/generation/vllm_generator.py`
5. `src/generation/post_processor.py`
6. `src/generation/rag_pipeline.py`

### Configuration
- `.env.example` (updated with generation settings)
- `src/config/settings.py` (extended with generation config)
- `requirements.txt` (added vllm, tiktoken, aiohttp)

### Examples (5 files)
1. `examples/04_setup_vllm.sh`
2. `examples/05_test_generation.py`
3. `examples/06_test_streaming.py`
4. `examples/07_conversation_demo.py`
5. `examples/08_complete_rag_demo.py`

---

**Total:** 14 new/modified files for Phase 2A
