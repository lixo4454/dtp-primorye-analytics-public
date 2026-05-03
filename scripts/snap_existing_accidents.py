"""
Одноразовый batch-скрипт: snap-to-road для всех исторических ДТП.

Что делает:
1. Берёт все accidents WHERE point IS NOT NULL AND point_snapped IS NULL
   AND snap_method IS NULL.
2. Обрабатывает батчами по 1000 через src.analysis.snap_to_road.snap_batch_sql.
3. Идемпотентно: повторный запуск обработает только не-snapped'ы.
4. По окончании — печатает sanity-distribution (median / p90 / p99
   расстояний, % unchanged).
5. Сохраняет гистограмму в data/processed/snap_distance_distribution.png

Запуск (один раз после build_osm_road_graph.py):
    python -m scripts.snap_existing_accidents

Время на 28k записей: ~2-3 мин на машине автора (зависит от объёма
osm_roads и нагрузки на Postgres).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.snap_to_road import snap_batch_sql, stats  # noqa: E402
from src.database.session import SessionLocal  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("snap_existing")

OUT_PNG = ROOT / "data" / "processed" / "snap_distance_distribution.png"


def main() -> None:
    logger.info("=" * 70)
    logger.info("Snap-to-road batch для исторических ДТП")
    logger.info("=" * 70)

    # Pre-check: сколько роадов в БД?
    with SessionLocal() as s:
        n_roads = s.execute(text("SELECT COUNT(*) FROM osm_roads")).scalar() or 0
        if n_roads == 0:
            raise RuntimeError(
                "osm_roads пусто — сначала запусти `python -m scripts.build_osm_road_graph`"
            )
        logger.info(f"osm_roads.count = {n_roads}")
        n_pending = (
            s.execute(
                text("""
            SELECT COUNT(*) FROM accidents
            WHERE point IS NOT NULL
              AND point_snapped IS NULL
              AND snap_method IS NULL
        """)
            ).scalar()
            or 0
        )
        logger.info(f"к обработке: {n_pending}")

    if n_pending == 0:
        logger.info("Все точки уже обработаны")
    else:
        total = {"osm_road": 0, "unchanged": 0}
        t0 = time.perf_counter()
        iter_no = 0
        while True:
            with SessionLocal() as s:
                counters = snap_batch_sql(s, batch_size=1000)
            iter_no += 1
            for k in total:
                total[k] += counters.get(k, 0)
            processed = sum(counters.values())
            logger.info(f"[batch {iter_no:3d}] {counters} → cumul {total}")
            if processed == 0:
                break
        dt = time.perf_counter() - t0
        logger.info(f"Готово за {dt:.1f}s, итог: {total}")

    # Финальная статистика + гистограмма
    with SessionLocal() as s:
        st = stats(s)
        logger.info(f"method_counts: {st['method_counts']}")
        logger.info(f"distance_stats: {st['snapped_distance_stats']}")

        # Гистограмма snap_distance_m по успешно snapped'ам
        rows = s.execute(
            text("""
            SELECT snap_distance_m FROM accidents
            WHERE snap_method = 'osm_road'
        """)
        ).all()
        distances = np.array([float(r[0]) for r in rows if r[0] is not None])

    if distances.size == 0:
        logger.warning("Нет snap'нутых точек, гистограмма не строится")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(distances, bins=50, color="#3b8bba", edgecolor="white", alpha=0.85)
    ax.axvline(
        np.median(distances),
        color="red",
        linestyle="--",
        label=f"медиана {np.median(distances):.1f} м",
    )
    ax.axvline(
        np.percentile(distances, 90),
        color="orange",
        linestyle="--",
        label=f"p90 {np.percentile(distances, 90):.1f} м",
    )
    ax.set_xlabel("snap_distance, м")
    ax.set_ylabel("количество ДТП")
    ax.set_title(
        f"Распределение расстояний raw → snapped point\n"
        f"(N={len(distances)}, max={distances.max():.0f} м)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    logger.info(f"Histogram saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
