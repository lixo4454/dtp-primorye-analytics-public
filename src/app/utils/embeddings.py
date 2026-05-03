"""Тонкий клиент для FastAPI ``/nlp/search``.

Изначально был задуман in-process semantic
search — sentence-transformers загружались прямо в Streamlit. После
ревью переехали на FastAPI-эндпоинт ``/nlp/search``: контейнер api уже
содержит torch + sentence-transformers (для bertopic), а dtp_app
остаётся ~331 МБ без ML-стека. Это и проще, и архитектурно чище.

Здесь только обёртка с кешированием — query → top-k результатов.
"""

from __future__ import annotations

import httpx
import streamlit as st

from src.app.utils.api_client import API_BASE_URL


@st.cache_data(ttl=600, show_spinner=False)
def semantic_search(query: str, top_k: int = 5) -> dict:
    """Cached-вызов /nlp/search. Кеш по (query, top_k) — повторный
    ввод того же запроса не дёргает sentence-transformer повторно."""
    if not query.strip():
        return {"items": [], "elapsed_ms": 0, "query": query, "top_k": top_k}
    with httpx.Client(base_url=API_BASE_URL, timeout=30.0) as client:
        r = client.get("/nlp/search", params={"q": query, "top_k": top_k})
        r.raise_for_status()
        return r.json()
