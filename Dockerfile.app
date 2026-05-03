## Production-grade образ для Streamlit-дашборда.
##
## Минимальный набор зависимостей — `requirements-app.txt`:
## - streamlit + folium + plotly — UI-стек
## - httpx — клиент к FastAPI
## - sqlalchemy + psycopg + pandas — direct DB для тяжёлых агрегатов
##
## Что НЕ ставим (отделяем от api / training-образов):
## - torch / catboost / prophet / bertopic / sentence-transformers — ML-инференс в API
## - natasha / pymorphy3 — NLP в pipeline'е, не в дашборде
## - cmdstan — Stan-бэкенд только для Prophet-инференса
## - geoalchemy2 / shapely — PostGIS-объекты в src.database.models, дашборду не нужны
##
## Запуск:
##   docker compose up -d app
## Доступ:
##   http://localhost:8501
## Healthcheck:
##   curl http://localhost:8501/_stcore/health  → "ok"

FROM python:3.12-slim

# Системные зависимости: build-essential для wheels (если pip не найдёт
# pre-built), libpq-dev для psycopg, curl для healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Минимальный requirements-app.txt вместо общего requirements.txt —
# образ получается ~600 МБ vs 3+ГБ с torch/catboost/bertopic.
COPY requirements-app.txt /app/requirements-app.txt
RUN pip install --no-cache-dir -r requirements-app.txt

# Копируем только то, что нужно дашборду:
# - src/app/    — сам дашборд
# - .streamlit/ — тема и server-config
# Остальные модули (src/database, src/ml, src/nlp, src/api) НЕ копируем,
# чтобы образ не зависел от их кода и pip-зависимостей.
COPY src/app/ /app/src/app/
COPY .streamlit/ /app/.streamlit/

# Маленькие data-артефакты, нужные strictly для UI-страниц:
# - catboost_form_schema.json (8.6 КБ): дефолты severity-формы
# - telegram_umap_2d.npz       (~75 КБ): UMAP scatter на NLP-странице
# Без них severity/nlp страницы 500'нут на cold-start. ML-pickle'и
# (catboost_features.pkl 5 МБ, telegram_embeddings.npz 3 МБ и т.д.)
# СОЗНАТЕЛЬНО НЕ копируем — они нужны только api-контейнеру.
COPY data/processed/catboost_form_schema.json /app/data/processed/catboost_form_schema.json
COPY data/processed/telegram_umap_2d.npz /app/data/processed/telegram_umap_2d.npz

# Не-root пользователь — стандартная hardening-практика
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Healthcheck — Streamlit отдаёт "ok" на /_stcore/health (внутренний путь)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

# Production-команда: один процесс на контейнер. Горизонтальное
# масштабирование — через `deploy.replicas` (Streamlit stateful по
# session_state, sticky-сессии нужны на reverse-proxy уровне).
CMD ["streamlit", "run", "src/app/main.py", \
     "--server.headless=true", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
