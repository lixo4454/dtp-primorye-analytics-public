# DTP Primorye Analytics — анализ ДТП Приморского края

[![CI](https://github.com/lixo4454/dtp-primorye-analytics/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lixo4454/dtp-primorye-analytics/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v1.0.0-1f4e79.svg)](#)

Программа для интеллектуального анализа дорожно-транспортных происшествий
и поддержки принятия решений (СППР) по безопасности дорожного движения
в Приморском крае. **29 413 ДТП** из открытых данных ГИБДД 2015–2026 года
+ **2 122 текстовых сводки** канала **«Полиция Приморья» (@prim_police)**
с автоматической связкой через NLP.

## Что делает программа

1. **Загружает и инкрементально обновляет** данные ГИБДД из dtp-stat.ru
   (Celery beat по понедельникам в 03:00 UTC+10).
2. **Парсит** текстовые сводки Telegram-канала **«Полиция Приморья»**
   (`@prim_police`, 2018–2025) и связывает их с записями ГИБДД через
   matcher (день/час/адрес/жертвы) — собрано **482 gold-пары**
   с top_score ≥ 75.
3. **Кластеризует очаги аварийности** методом DBSCAN
   (eps = 300 м, min_samples = 15) — найдены **120 очагов**, топ-30
   визуализированы на интерактивной карте.
4. **Прогнозирует** количество ДТП помесячно через Prophet
   с COVID-регрессором; на hold-out 2025 — **MAPE 5.4 %**.
5. **Классифицирует тяжесть** конкретного ДТП через CatBoost v2
   (Optuna, 200 trials) + isotonic-калибровку; **ECE < 0.05**
   для всех 4 классов (`light` / `severe` / `severe_multiple` / `dead`).
6. **Тематически кластеризует** Telegram-посты через BERTopic
   (sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2`) —
   **7 содержательных тем** + шум.
7. **Извлекает именованные сущности** (адреса, возраст, марки ТС)
   через Natasha + pymorphy3 — **30 % покрытие** возрастом водителя.
8. **Выдаёт рекомендации** — система поддержки принятия решений с
   **18 правилами** (R01–R18: speed-30, cable barrier, HFST,
   anti-icing brine и др.) с evidence base из научных публикаций
   и counterfactual-симуляцией мер.
9. **Snap-to-road** — координаты ДТП проецируются на ребра OSM-графа
   (192 934 рёбер для Приморья); покрытие 87.5 % (median смещение 2.1 м).

## Главные находки

- **Гипотеза МВД о 4.9× опасности праворульных авто (RHD) — отвергнута.**
  Контроль по категории ТС, возрасту ТС и типу местности даёт обратное:
  RHD на **35 % безопаснее** LHD (ratio 0.65×, Wald-test p < 0.001).
  Гипотеза МВД методологически некорректна — не учитывала confounders.
- **Дефект T4 в Telegram-фильтре:** rule-based `is_dtp_related` пропускал
  72 не-ДТП поста (юбилеи, кадровые); BERTopic post-hoc обнаружил.
- **Дефект `model` в GeoJSON источника:** поле `vehicle.model` содержит
  марку, реальная модель — в `vehicle.color` (sic!). Решение —
  собственный справочник 199 марок brand → RHD/LHD.

## Технологический стек

| Слой | Технологии |
|---|---|
| Язык | Python 3.12 |
| База данных | PostgreSQL 16 + PostGIS, SQLAlchemy 2.0, Alembic |
| Очередь и кеш | Redis 7, Celery (4 beat-задачи) |
| Backend API | FastAPI + Pydantic, 11 эндпоинтов, Swagger UI |
| Frontend | Streamlit 1.41 + Folium + Plotly, 7 multipage страниц |
| Машинное обучение | scikit-learn, CatBoost (Optuna 200 trials), Prophet, isotonic calibration |
| NLP (русский) | Natasha, pymorphy3, sentence-transformers, BERTopic, UMAP, HDBSCAN |
| Контейнеризация | Docker + docker-compose (6 сервисов) |
| Тесты | pytest (125 unit + integration smoke), GitHub Actions CI |
| Качество кода | Ruff (lint + format), pre-commit |

## Источники данных

- **[dtp-stat.ru](https://dtp-stat.ru/)** — независимый агрегатор открытых
  данных МВД РФ. Скачиваем единый архив 2015-2026 по всей России
  (`https://dtp-stat.ru/media/years/2015-2026.json.7z`), затем фильтруем
  по сухопутному полигону Приморского края через PostGIS. Еженедельное
  автообновление через Celery beat (по понедельникам в 03:00 UTC+10).
  Первоисточником данных в dtp-stat.ru являются официальные карточки
  ГИБДД (в т. ч. сайт МВД 25 региона — [25.мвд.рф](https://25.мвд.рф)).
- **Telegram-канал [«Полиция Приморья» (@prim_police)](https://t.me/prim_police)** —
  текстовые сводки оперативной обстановки от полиции края (~22 000
  публичных постов, ~2 122 относятся к ДТП). Загружаются ручным экспортом
  Telegram Desktop (`result.json`) → `src/loaders/telegram_export_loader.py`.

## Запуск

Минимум — Docker Desktop (4 ГБ RAM). Stack поднимается одной командой:

```bash
docker compose up -d
```

После healthcheck (~30–60 секунд) откроются:

- **Дашборд:** <http://localhost:8501>
- **API + Swagger UI:** <http://localhost:8000/docs>

Postgres и Redis слушают `5432` и `6379` локально (для удобства подключения
из IDE/dbeaver).

### Dev режим vs prod

`docker-compose.override.yml` применяется автоматически и монтирует
`./src` + `./.streamlit` как read-only volumes — изменения подхватываются
без `docker compose build` (Streamlit `runOnSave`, uvicorn `--reload`).

Для production-сборки (без override):

```bash
docker compose -f docker-compose.yml up -d
```

## Разработка

### Установка локального окружения

```bash
python -m venv venv
.\venv\Scripts\activate.ps1     # PowerShell
pip install -r requirements.txt
pre-commit install              # автохуки на каждый git commit
```

### Тесты

```bash
# unit (125 тестов, без БД, ~2 сек)
pytest tests/unit -v

# coverage
pytest tests/unit --cov=src.loaders --cov=src.nlp --cov=src.analysis

# integration (нужен docker compose up)
pytest tests/test_api_smoke.py tests/test_day17_admin.py -v
```

### Линтер и форматтер

```bash
ruff check src tests       # lint
ruff check src tests --fix # auto-fix
ruff format src tests      # format (как Black)
```

Конфиг — в [`pyproject.toml`](pyproject.toml). Pre-commit хуки —
в [`.pre-commit-config.yaml`](.pre-commit-config.yaml).

### CI

GitHub Actions запускает 3 параллельных job'а на каждом push/PR в `main`:

1. **lint** — `ruff check`
2. **test** — pytest unit-тесты (без heavy ML-зависимостей)
3. **build** — сборка Docker-образа `dtp_api`

Конфиг — [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Структура репозитория

```
.
├── .streamlit/             # config.toml (палитра + шрифты) + style.css
├── .github/workflows/      # CI: lint + test + build
├── alembic/                # миграции схемы БД
├── data/
│   ├── raw/                # GeoJSON выгрузки ГИБДД (gitignored)
│   └── processed/          # промежуточные файлы (gitignored, регенерируются)
├── models/                 # натренированные .cbm / .pkl (gitignored)
├── scripts/                # CLI-скрипты (parser, retrain)
├── src/
│   ├── analysis/           # DBSCAN, RHD/LHD, severity, recommendations rule engine
│   ├── api/                # FastAPI: routers, schemas, dependencies
│   ├── app/                # Streamlit: views/, utils/ (api_client, db, styling)
│   ├── database/           # SQLAlchemy models + Alembic env
│   ├── loaders/            # парсеры dtp-stat.ru, миграции данных, snap-to-road
│   ├── ml/                 # CatBoost + Prophet pipelines, calibration
│   ├── nlp/                # Natasha + matcher Telegram ↔ ГИБДД, BERTopic
│   └── tasks/              # Celery beat задачи (parser, retrain, refresh)
└── tests/
    ├── unit/               # 125 unit-тестов (pure функции)
    └── test_*_smoke.py     # integration smoke (требует docker compose)
```

## Лицензия

[MIT](LICENSE) — Лазакович Алексей Евгеньевич, 2026.
