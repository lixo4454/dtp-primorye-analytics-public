"""Streamlit-дашборд для проекта анализа ДТП Приморского края.

Запуск из корня проекта:
    streamlit run src/app/main.py

Архитектура:
- ``main.py`` — точка входа со ``st.navigation`` (modern multipage API ≥1.36)
- ``views/`` — страницы (home, hotspots_map, dtp_types)
- ``utils/api_client.py`` — sync httpx-клиент к FastAPI с кешированием
- ``utils/db.py`` — прямые SQL-запросы для тяжёлых агрегатов
- ``utils/visualizations.py`` — общие константы и стили графиков
"""
