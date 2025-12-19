"""
Simplified Semantic Cache Wrapper

This module provides a clean, simplified interface for semantic caching with optional reranking.

Example usage:
    from cache.wrapper import SemanticCacheWrapper

    # Create cache wrapper from config
    cache = SemanticCacheWrapper.from_config(config)

    # Or create with custom parameters
    cache = SemanticCacheWrapper(
        name="my-cache",
        distance_threshold=0.3,
        ttl=3600
    )

    # Hydrate cache from DataFrame
    cache.hydrate_from_df(df, q_col="question", a_col="answer")

    # Check cache
    results = cache.check("What is your refund policy?")

    # Check multiple queries
    results = cache.check_many(queries, show_progress=True)
"""

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import redis
from pydantic import BaseModel
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
from scipy.spatial.distance import cosine
from tqdm.auto import tqdm

from cache.config import config as default_config


class CacheResult(BaseModel):
    """
    Standardized result for cache wrapper outputs.

    Core Fields:
    - prompt: cache key text
    - response: cached response text
    - vector_distance: semantic distance from vector index (lower = more similar)
    - cosine_similarity: cosine similarity score (higher = more similar)

    Reranker Metadata (optional):
    - reranker_type: type of reranker used
    - reranker_score: raw score from reranker
    - reranker_reason: explanation from reranker
    """

    prompt: str
    response: str
    vector_distance: float
    cosine_similarity: float

    reranker_type: Optional[str] = None
    reranker_score: Optional[float] = None
    reranker_reason: Optional[str] = None


class CacheResults(BaseModel):
    """Container for cache check results."""
    query: str
    matches: List[CacheResult]

    def __repr__(self):
        return f"(Query: '{self.query}', Matches: {[m.prompt for m in self.matches]})"


def try_connect_to_redis(redis_url: str):
    """Test Redis connection and return client."""
    try:
        r = redis.Redis.from_url(redis_url)
        r.ping()
        print("Redis is running and accessible!")
    except redis.ConnectionError:
        print(
            """
            Cannot connect to Redis. Please make sure Redis is running on localhost:6379
                Try: docker run -d --name redis -p 6379:6379 redis/redis-stack:latest
            """
        )
        raise

    return r


class SemanticCacheWrapper:
    """
    A wrapper around RedisVL SemanticCache that provides:
    - Easy configuration from dict/config objects
    - DataFrame and pair-based cache hydration
    - Batch checking with progress bars
    - Optional reranker support
    """

    def __init__(
        self,
        name: str = "semantic-cache",
        distance_threshold: float = 0.3,
        ttl: int = 3600,
        redis_url: Optional[str] = None,
    ):
        redis_conn_url = redis_url or default_config.get(
            "redis_url", "redis://localhost:6379"
        )
        self.redis = try_connect_to_redis(redis_conn_url)

        self.embeddings_cache = EmbeddingsCache(redis_client=self.redis, ttl=ttl * 24)
        self.langcache_embed = HFTextVectorizer(
            model="redis/langcache-embed-v1", cache=self.embeddings_cache
        )
        self.cache = SemanticCache(
            name=name,
            vectorizer=self.langcache_embed,
            redis_client=self.redis,
            distance_threshold=distance_threshold,
            ttl=ttl,
        )

        self._reranker: Optional[Callable[[str, List[dict]], List[dict]]] = None

    def pair_distance(self, question: str, answer: str) -> float:
        """Compute semantic distance between question and answer."""
        q_emb = self.langcache_embed.embed(question)
        a_emb = self.langcache_embed.embed(answer)
        distance = cosine(q_emb, a_emb)
        return distance.item()

    def set_cache_entries(self, question_answer_pairs: List[Tuple[str, str]]):
        """Clear cache and set new entries from Q&A pairs."""
        self.cache.clear()
        for question, answer in question_answer_pairs:
            self.cache.store(prompt=question, response=answer)

    @classmethod
    def from_config(cls, config) -> "SemanticCacheWrapper":
        """
        Construct a SemanticCacheWrapper from a config dict.

        Expected config keys:
        - redis_url (default: "redis://localhost:6379")
        - cache_name (default: "semantic-cache")
        - distance_threshold (default: 0.3)
        - ttl_seconds (default: 3600)
        """
        return cls(
            redis_url=config.get("redis_url", "redis://localhost:6379"),
            name=config.get("cache_name", "semantic-cache"),
            distance_threshold=float(config.get("distance_threshold", 0.3)),
            ttl=int(config.get("ttl_seconds", 3600)),
        )

    def hydrate_from_df(
        self,
        df: pd.DataFrame,
        *,
        q_col: str = "question",
        a_col: str = "answer",
        clear: bool = True,
        ttl_override: Optional[int] = None,
        return_id_map: bool = False,
    ) -> Optional[Dict[str, int]]:
        """
        Populate cache from a DataFrame.

        Args:
            df: DataFrame with question and answer columns
            q_col: Name of question column
            a_col: Name of answer column
            clear: Whether to clear existing cache first
            ttl_override: Optional TTL override for these entries
            return_id_map: Whether to return mapping of questions to IDs
        """
        if clear:
            self.cache.clear()
        question_to_id: Dict[str, int] = {}
        idx = 0
        for row in df[[q_col, a_col]].itertuples(index=False, name=None):
            q, a = row
            self.cache.store(prompt=q, response=a, ttl=ttl_override)
            if return_id_map and q not in question_to_id:
                question_to_id[q] = idx
            idx += 1
        return question_to_id if return_id_map else None

    def hydrate_from_pairs(
        self,
        pairs: Iterable[Tuple[str, str]],
        *,
        clear: bool = True,
        ttl_override: Optional[int] = None,
        return_id_map: bool = False,
    ) -> Optional[Dict[str, int]]:
        """
        Populate cache from iterable of (question, answer) pairs.
        """
        if clear:
            self.cache.clear()
        question_to_id: Dict[str, int] = {}
        idx = 0
        for q, a in pairs:
            self.cache.store(prompt=q, response=a, ttl=ttl_override)
            if return_id_map and q not in question_to_id:
                question_to_id[q] = idx
            idx += 1
        return question_to_id if return_id_map else None

    def register_reranker(self, reranker: Callable[[str, List[dict]], List[dict]]):
        """
        Register a reranking function.

        The reranker should have signature:
        reranker(query: str, candidates: List[dict]) -> List[dict]
        """
        if not callable(reranker):
            raise TypeError("Reranker must be a callable function")
        self._reranker = reranker

    def clear_reranker(self):
        """Remove any registered reranking function."""
        self._reranker = None

    def has_reranker(self) -> bool:
        """Check if a reranker is registered."""
        return self._reranker is not None

    def check(
        self,
        query: str,
        distance_threshold: Optional[float] = None,
        num_results: int = 1,
        use_reranker_distance: bool = False,
    ) -> "CacheResults":
        """
        Check semantic cache for a single query.

        Args:
            query: The query string to search for
            distance_threshold: Maximum semantic distance (lower = more similar)
            num_results: Maximum number of results to return

        Returns:
            CacheResults object with matches
        """
        _num_results = (
            num_results if not self.has_reranker() else max(10, 3 * num_results)
        )
        candidates = self.cache.check(
            query, distance_threshold=distance_threshold, num_results=_num_results
        )

        if not candidates:
            return CacheResults(query=query, matches=[])

        if self.has_reranker():
            candidates = self._reranker(query, candidates)

        results: List[CacheResult] = []
        for item in candidates[:num_results]:
            result = dict(item)
            result["vector_distance"] = float(result.get("vector_distance", 0.0))
            result["cosine_similarity"] = float((2 - result["vector_distance"]) / 2)
            result["query"] = query

            if self.has_reranker():
                result["reranker_type"] = result.get("reranker_type")
                result["reranker_score"] = result.get("reranker_score")
                result["reranker_reason"] = result.get("reranker_reason")
                if use_reranker_distance:
                    result["vector_distance"] = result.get("reranker_distance", result["vector_distance"])

            results.append(CacheResult(**result))

        return CacheResults(query=query, matches=results)

    def check_many(
        self,
        queries: List[str],
        distance_threshold: Optional[float] = None,
        show_progress: bool = False,
        num_results: int = 1,
        use_reranker_distance: bool = False,
    ) -> List["CacheResults"]:
        """
        Check semantic cache for multiple queries.

        Args:
            queries: List of query strings
            distance_threshold: Maximum semantic distance
            show_progress: Whether to show progress bar
            num_results: Maximum results per query

        Returns:
            List of CacheResults (maintains query order)
        """
        results: List[CacheResults] = []
        for q in tqdm(queries, disable=not show_progress):
            cache_results = self.check(
                q, distance_threshold, num_results, use_reranker_distance
            )
            results.append(cache_results)
        return results

    def store(self, prompt: str, response: str, **kwargs):
        """Store a prompt-response pair in the cache."""
        self.cache.store(prompt=prompt, response=response, **kwargs)

    def clear(self):
        """Clear all entries from the cache."""
        self.cache.clear()
