# Redis Semantic Caching - Documentation

This document provides reference documentation for the helper utilities and key concepts used in the Redis Semantic Caching project. All information is extracted from course materials.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [SemanticCacheWrapper](#semanticcachewrapper)
4. [CacheEvaluator](#cacheevaluator)
5. [PerfEval](#perfeval)
6. [Key Concepts](#key-concepts)
7. [Common Patterns](#common-patterns)
8. [FAQ Data Format](#faq-data-format)

---

## Installation

```bash
pip install sentence-transformers pandas numpy redisvl langchain-openai langchain-core
pip install python-dotenv redis tiktoken
```

Start Redis:
```bash
docker run -d --name redis -p 6379:6379 redis/redis-stack:latest
```

---

## Configuration

### cache.config

```python
from cache.config import config, load_openai_key

# Config dict with defaults from environment
config = {
    "redis_url": "redis://localhost:6379",  # REDIS_URL env var
    "cache_name": "semantic-cache",          # CACHE_NAME env var
    "distance_threshold": 0.3,               # CACHE_DISTANCE_THRESHOLD env var
    "ttl_seconds": 3600                      # CACHE_TTL_SECONDS env var
}

# Load OpenAI API key from environment or prompt
load_openai_key()
```

### Environment Variables

Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
REDIS_URL=redis://localhost:6379
CACHE_NAME=semantic-cache
CACHE_DISTANCE_THRESHOLD=0.3
CACHE_TTL_SECONDS=3600
```

---

## SemanticCacheWrapper

A simplified interface for semantic caching with Redis. Uses `redis/langcache-embed-v1` embedding model.

### Creating a Cache

```python
from cache.wrapper import SemanticCacheWrapper
from cache.config import config

# Option 1: From config
cache = SemanticCacheWrapper.from_config(config)

# Option 2: Custom parameters
cache = SemanticCacheWrapper(
    name="my-cache",
    distance_threshold=0.3,
    ttl=3600,
    redis_url="redis://localhost:6379"
)
```

### Hydrating the Cache

Load data into the cache (clears existing entries by default):

```python
import pandas as pd

# From DataFrame
df = pd.DataFrame({
    "question": ["How do I get a refund?", "Can I reset my password?"],
    "answer": ["Visit your orders page...", "Click Forgot Password..."]
})
cache.hydrate_from_df(df, q_col="question", a_col="answer")

# From list of pairs
pairs = [
    ("How do I get a refund?", "Visit your orders page..."),
    ("Can I reset my password?", "Click Forgot Password...")
]
cache.hydrate_from_pairs(pairs)

# Without clearing existing cache
cache.hydrate_from_df(df, clear=False)
```

### Checking the Cache

```python
# Single query - returns CacheResults object
results = cache.check("How can I get my money back?")

if results.matches:
    match = results.matches[0]
    print(f"Hit: {match.prompt}")
    print(f"Response: {match.response}")
    print(f"Distance: {match.vector_distance}")
    print(f"Similarity: {match.cosine_similarity}")
else:
    print("Cache miss")

# Multiple queries with progress bar
results_list = cache.check_many(
    queries=["query1", "query2", "query3"],
    distance_threshold=0.3,
    show_progress=True,
    num_results=1
)

# Custom distance threshold
results = cache.check("query", distance_threshold=0.2)
```

### Storing Entries

```python
# Store a single entry
cache.store(prompt="user question", response="llm response")

# Clear all entries
cache.clear()

# Set multiple entries (clears first)
cache.set_cache_entries([
    ("question1", "answer1"),
    ("question2", "answer2")
])
```

### CacheResult and CacheResults Objects

```python
# CacheResults contains query and list of matches
class CacheResults:
    query: str
    matches: List[CacheResult]

# Each match is a CacheResult
class CacheResult:
    prompt: str              # Cached question
    response: str            # Cached response
    vector_distance: float   # Lower = more similar (0.0 = identical)
    cosine_similarity: float # Higher = more similar (1.0 = identical)
```

### Complete Example from Course

```python
from cache.wrapper import SemanticCacheWrapper
from cache.config import config

# Initialize
cache = SemanticCacheWrapper.from_config(config)

# Load FAQ data
import pandas as pd
faq_df = pd.DataFrame({
    "question": [
        "How do I get a refund?",
        "Can I reset my password?",
        "Where is my order?",
        "How long is the warranty?",
        "Do you ship internationally?",
        "How do I cancel my subscription?",
        "Can I change my delivery address?",
        "What payment methods do you accept?"
    ],
    "answer": [
        "To request a refund, visit your orders page and select **Request Refund**. Refunds are processed within 3-5 business days.",
        "Click **Forgot Password** on the login page and follow the email instructions. Contact support if you don't receive the email within 10 minutes.",
        "Use the tracking link sent to your email after purchase. Allow 24-48 hours for tracking to activate.",
        "All electronic products include a 12-month warranty. Extended warranties available at checkout.",
        "Yes, we ship to over 50 countries worldwide. International shipping typically takes 7-14 business days.",
        "Go to Account Settings > Subscriptions and click **Cancel Subscription**. You'll retain access until the end of your billing period.",
        "Yes, you can update your delivery address in Account Settings > Addresses before your order ships.",
        "We accept all major credit cards, PayPal, and Apple Pay. Some regions also support local payment methods."
    ]
})

# Hydrate cache
cache.hydrate_from_df(faq_df)

# Check cache
results = cache.check("I want my money back")
if results.matches:
    print(f"Cache hit: {results.matches[0].response}")
```

---

## CacheEvaluator

Evaluate cache effectiveness with precision, recall, and F1 metrics.

### Basic Usage

```python
from cache.evals import CacheEvaluator

# Create evaluator with true labels and cache results
evaluator = CacheEvaluator(
    true_labels=[True, True, False, True, False],  # Ground truth
    cache_results=cache_results  # List of CacheResults from check_many()
)

# Get metrics at default threshold
metrics = evaluator.get_metrics()
print(f"Hit Rate: {metrics['cache_hit_rate']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
print(f"F1 Score: {metrics['f1_score']:.2%}")
print(f"Accuracy: {metrics['accuracy']:.2%}")

# Get metrics at specific threshold
metrics = evaluator.get_metrics(distance_threshold=0.25)
```

### From Full Retrieval

When evaluating with full retrieval (no threshold filtering):

```python
# Get all matches regardless of threshold
cache_results = cache.check_many(queries, distance_threshold=1.0)

# Create evaluator for full retrieval
evaluator = CacheEvaluator.from_full_retrieval(
    true_labels=true_labels,
    cache_results=cache_results
)
```

### Threshold Sweep

Find the optimal threshold:

```python
# Sweep thresholds to maximize F1 score
sweep_results = evaluator.sweep_thresholds(
    metric_to_maximize="f1_score",
    threshold_range=(0, 1),
    num_samples=100
)

print(f"Best threshold: {sweep_results['best_threshold']:.3f}")

# Access all metrics across thresholds
all_metrics = sweep_results["all_metrics"]
thresholds = sweep_results["thresholds"]
```

### View Matches DataFrame

```python
# Get DataFrame of query-match-distance-label
df = evaluator.matches_df()
print(df.head())
# Columns: query, match, distance, true_label
```

### Metrics Returned

```python
{
    "cache_hit_rate": float,    # (TP + FP) / Total
    "precision": float,         # TP / (TP + FP)
    "recall": float,            # TP / (TP + FN)
    "f1_score": float,          # Harmonic mean of precision & recall
    "accuracy": float,          # (TP + TN) / Total
    "utility": float,           # Harmonic mean of precision & hit rate
    "confusion_matrix": array   # [[TN, FP], [FN, TP]]
}
```

---

## PerfEval

Track latency, throughput, and LLM costs.

### Basic Usage

```python
from cache.evals import PerfEval

perf = PerfEval()
perf.set_total_queries(len(queries))

with perf:
    for query in queries:
        perf.start()
        results = cache.check(query)

        if results.matches:
            perf.tick("cache_hit")
            response = results.matches[0].response
        else:
            perf.tick("cache_miss")

            # Call LLM
            perf.start()
            response = llm(query)
            perf.tick("llm_call")

            # Record for cost tracking
            perf.record_llm_call("gpt-4o-mini", query, response)

            cache.store(prompt=query, response=response)

# Get metrics
metrics = perf.get_metrics(labels=["cache_hit", "llm_call"])
costs = perf.get_costs()

# Print summary
print(perf.summary(labels=["cache_hit", "llm_call"]))
```

### Metrics Structure

```python
metrics = perf.get_metrics(labels=["cache_hit", "llm_call"])

# Overall metrics
print(metrics["overall"]["average_latency_ms"])
print(metrics["overall"]["p50_ms"])
print(metrics["overall"]["p95_ms"])

# By label
print(metrics["by_label"]["cache_hit"]["count"])
print(metrics["by_label"]["cache_hit"]["average_latency_ms"])
print(metrics["by_label"]["llm_call"]["average_latency_ms"])
```

### Cost Tracking

```python
# Record LLM calls
perf.record_llm_call(
    model="gpt-4o-mini",
    input_text="user query",
    output_text="llm response",
    provider="openai"
)

# Get costs
costs = perf.get_costs()
print(f"Total Cost: ${costs['total_cost']:.4f}")
print(f"Calls: {costs['calls']}")
print(f"Avg Cost/Query: ${costs['avg_cost_per_query']:.6f}")
print(f"By Model: {costs['by_model']}")
```

### Simulating LLM Calls

From course example:
```python
import time
import numpy as np

def simulate_llm_call(prompt):
    time.sleep(np.random.uniform(0.2, 0.5))
    return f"LLM response to {prompt}"
```

---

## Key Concepts

### Distance vs Similarity

- **Vector Distance**: Lower = more similar (0.0 = identical, 2.0 = opposite for cosine)
- **Cosine Similarity**: Higher = more similar (1.0 = identical, 0.0 = orthogonal)

Conversion: `similarity = (2 - distance) / 2`

### Threshold Selection

| Threshold | Similarity | Behavior |
|-----------|------------|----------|
| 0.05-0.10 | 95-97% | Very strict, high precision |
| 0.10-0.15 | 92-95% | Balanced (recommended) |
| 0.15-0.25 | 87-92% | Loose, high hit rate |
| 0.25-0.30 | 85-87% | Default, good starting point |

### Precision vs Recall Trade-off

- **Precision**: How many cache hits were correct?
- **Recall**: How many correct matches did we find?
- **F1 Score**: Balanced metric (harmonic mean)

Lower threshold → Higher precision, lower recall
Higher threshold → Lower precision, higher recall

### Cache Hit Rate vs Utility

- **Hit Rate**: Fraction of queries that hit cache
- **Utility**: Harmonic mean of precision and hit rate

Optimize for utility to balance cost savings (hit rate) with accuracy (precision).

---

## Common Patterns

### Cache-or-Generate Pattern

```python
def get_response(query: str, llm_fn) -> str:
    results = cache.check(query)

    if results.matches:
        return results.matches[0].response

    response = llm_fn(query)
    cache.store(prompt=query, response=response)
    return response
```

### End-to-End LLM Example

From L2 course notebook:

```python
from cache.config import load_openai_key
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_openai_key()

MODEL_NAME = "gpt-4o-mini"

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=150,
)

def get_llm_response(question: str) -> str:
    prompt = f"""
    You are a helpful customer support assistant. Answer this customer question concisely and professionally:

    Question: {question}

    Provide a helpful response in 1-2 sentences. If you don't have specific information, give a general helpful response.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

# Usage with cache
from cache.evals import PerfEval

perf_eval = PerfEval()
test_questions = [
    "How can I get my money back?",
    "I want a refund please",
    "What's your return policy?",
    "I forgot my password",
]
perf_eval.set_total_queries(len(test_questions))

with perf_eval:
    for question in test_questions:
        perf_eval.start()

        if cached_result := cache.check(question):
            if cached_result.matches:
                perf_eval.tick("cache_hit")
                response = cached_result.matches[0].response
            else:
                perf_eval.tick("cache_miss")
                perf_eval.start()
                response = get_llm_response(question)
                perf_eval.tick("llm_call")
                perf_eval.record_llm_call(MODEL_NAME, question, response)
                cache.store(prompt=question, response=response)
```

### Evaluation Workflow

```python
# 1. Hydrate cache with seed data
cache.hydrate_from_df(faq_df)

# 2. Run queries and collect results
cache_results = cache.check_many(test_queries)

# 3. Create evaluator with ground truth
evaluator = CacheEvaluator(true_labels, cache_results)

# 4. Find optimal threshold
sweep = evaluator.sweep_thresholds(metric_to_maximize="f1_score")
best_threshold = sweep["best_threshold"]

# 5. Report final metrics
metrics = evaluator.get_metrics(distance_threshold=best_threshold)
```

### Performance Benchmarking

```python
perf = PerfEval()
perf.set_total_queries(len(test_queries))

with perf:
    for query in test_queries:
        perf.start()
        results = cache.check(query)

        if results.matches:
            perf.tick("cache_hit")
        else:
            perf.tick("cache_miss")
            perf.start()
            response = simulate_llm_call(query)
            perf.tick("llm_call")
            perf.record_llm_call("gpt-4o-mini", query, response)

# Calculate speedup
metrics = perf.get_metrics(labels=["cache_hit", "llm_call"])
cache_latency = metrics["by_label"]["cache_hit"]["average_latency_ms"]
llm_latency = metrics["by_label"]["llm_call"]["average_latency_ms"]
speedup = llm_latency / cache_latency
print(f"Cache is {speedup:.1f}x faster than LLM")
```

---

## FAQ Data Format

### Standard Format

The course uses this FAQ format:

```python
faq_df = pd.DataFrame({
    "question": [
        "How do I get a refund?",
        "Can I reset my password?",
        "Where is my order?",
        "How long is the warranty?",
        "Do you ship internationally?",
        "How do I cancel my subscription?",
        "Can I change my delivery address?",
        "What payment methods do you accept?"
    ],
    "answer": [
        "To request a refund, visit your orders page and select **Request Refund**. Refunds are processed within 3-5 business days.",
        "Click **Forgot Password** on the login page and follow the email instructions. Contact support if you don't receive the email within 10 minutes.",
        "Use the tracking link sent to your email after purchase. Allow 24-48 hours for tracking to activate.",
        "All electronic products include a 12-month warranty. Extended warranties available at checkout.",
        "Yes, we ship to over 50 countries worldwide. International shipping typically takes 7-14 business days.",
        "Go to Account Settings > Subscriptions and click **Cancel Subscription**. You'll retain access until the end of your billing period.",
        "Yes, you can update your delivery address in Account Settings > Addresses before your order ships.",
        "We accept all major credit cards, PayPal, and Apple Pay. Some regions also support local payment methods."
    ]
})
```

### Test Dataset Format

For evaluation, use a test dataset with labels:

```python
test_df = pd.DataFrame({
    "question": [
        "What's the process for getting my money back?",
        "How can I request a refund for my purchase?",
        "What's your refund policy for digital products?",
    ],
    "answer": [...],
    "src_question_id": [0, 0, 0],  # Maps to FAQ question
    "cache_hit": [True, True, False]  # Whether it should hit
})
```

---

## Model Costs Reference

| Model | Input ($/1K tokens) | Output ($/1K tokens) |
|-------|---------------------|----------------------|
| gpt-4o-mini | $0.00015 | $0.0006 |
| gpt-4o | $0.005 | $0.015 |
| gpt-3.5-turbo | $0.0005 | $0.0015 |

---

## Embedding Model

The default embedding model is `redis/langcache-embed-v1`, optimized for semantic caching.

Alternative models (from sentence-transformers):
- `all-MiniLM-L6-v2` (384 dims) - General purpose, fast
- `all-mpnet-base-v2` (768 dims) - Higher quality
- `multi-qa-MiniLM-L6-cos-v1` (384 dims) - Q&A optimized
