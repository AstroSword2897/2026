# Caching in MaxSight

This document describes how caching is used and how to configure it.

## Purpose

Caching reduces repeated work for the same inputs (e.g. model outputs, API responses, retrieval results). It is used where latency or cost matters and results are safe to reuse for a period.

## Redis cache

The main cache implementation is Redis-based in `ml/cache/redis_cache.py`.

- **Class:** `RedisCache`. Requires the `redis` package and a Redis server.
- **Configuration:** Pass `redis_url` or set the `REDIS_URL` environment variable (default: `redis://localhost:6379/0`). Optional `default_ttl` (seconds) for key expiry.
- **Operations:** `get(key)`, `set(key, value, ttl=None)`, `delete(key)`. Values are pickled for storage.
- **Use case:** Cache model outputs or external service responses keyed by input hash or request ID. TTL avoids stale data.

## When to use cache

- **Inference:** Caching full model outputs for repeated (e.g. identical) inputs can cut latency and load. Key design: key must reflect all inputs that affect the output (image, condition, model version, etc.).
- **Retrieval:** Index lookups or embedding computations can be cached by query or image ID when the index and model are fixed.
- **External APIs:** Any call to an external service (e.g. scene description, OCR) is a good candidate if responses are deterministic for the same input and you set a reasonable TTL.

## When not to cache

- **Training:** Do not cache training batches or gradients; data must change every epoch.
- **Real-time safety:** For safety-critical, real-time outputs (e.g. immediate hazard alerts), prefer fresh inference unless you have a clear invalidation and TTL strategy.
- **User-specific state:** Session state, therapy state, or per-user mutable data should live in a proper store (database or session backend), not only in a generic cache.

## Best practices

- Use a key scheme that includes model version and any relevant config so caches are invalidated when the model or config changes.
- Set TTLs so stale data does not persist indefinitely.
- Prefer small, serializable payloads; avoid caching very large tensors unless necessary.
- If Redis is unavailable, the code should degrade gracefully (e.g. skip cache and compute), not fail hard.

## Related code

- **Cache module:** `ml/cache/redis_cache.py`
- **Config:** `ml/config.py` may expose cache-related settings (e.g. enable/disable, TTL) if the app uses them.
- **Retrieval:** `ml/retrieval/` may use caching for index or query results; see retrieval docs for details.
