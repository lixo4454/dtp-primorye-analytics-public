"""
Idempotent populator for accidents.is_in_region via PostGIS ST_Within.

Что делает: одним UPDATE-запросом проставляет в `accidents.is_in_region`
значение `ST_Within(point, region_boundaries.geom)` для записей с
непустым `point`. По умолчанию обрабатывает только строки с
`is_in_region IS NULL` (идемпотентность); флаг `--force` пересчитывает
все строки заново.

Зачем нужно: материализует геопространственную проверку «координаты
ДТП лежат на суше Приморья?» в обычное BOOLEAN-поле для быстрых
последующих запросов (DBSCAN, агрегаты по районам, отчёты) — без
повторного вызова PostGIS-функции в каждом WHERE.
"""

from __future__ import annotations

import argparse
import time

from loguru import logger
from sqlalchemy import text

from src.database import SessionLocal


def update_is_in_region(region_name: str, force: bool) -> dict:
    """Запускает UPDATE и возвращает сводку по результатам."""
    with SessionLocal() as session:
        # Sanity: убедиться что регион существует в region_boundaries
        check = session.execute(
            text("SELECT id, area_km2 FROM region_boundaries WHERE region_name = :rn"),
            {"rn": region_name},
        ).first()
        if check is None:
            raise RuntimeError(
                f"Region {region_name!r} not found in region_boundaries — "
                f"run scripts/load_region_boundary.py first"
            )
        logger.info(
            "Region {!r} found: id={}, area_km2={}",
            region_name,
            check.id,
            check.area_km2,
        )

        # Сколько строк попадёт под обновление
        if force:
            target_clause = "WHERE point IS NOT NULL"
        else:
            target_clause = "WHERE point IS NOT NULL AND is_in_region IS NULL"

        cnt = session.execute(text(f"SELECT COUNT(*) FROM accidents {target_clause}")).scalar()
        logger.info(
            "Rows to update: {} (mode: {})",
            cnt,
            "force-recompute" if force else "fill NULL only",
        )

        if cnt == 0:
            logger.info("Nothing to do — all rows already have is_in_region set")
        else:
            t0 = time.time()
            result = session.execute(
                text(
                    f"""
                    UPDATE accidents
                    SET is_in_region = ST_Within(
                        point,
                        (SELECT geom FROM region_boundaries WHERE region_name = :rn)
                    )
                    {target_clause}
                    """
                ),
                {"rn": region_name},
            )
            session.commit()
            elapsed = time.time() - t0
            logger.info(
                "UPDATE complete: {} rows in {:.2f}s ({:.0f} rows/s)",
                result.rowcount,
                elapsed,
                result.rowcount / elapsed if elapsed > 0 else 0,
            )

        # Sanity-check: распределение значений после обновления
        dist = session.execute(
            text(
                """
                SELECT
                    COUNT(*) FILTER (WHERE is_in_region IS TRUE)  AS in_region,
                    COUNT(*) FILTER (WHERE is_in_region IS FALSE) AS out_region,
                    COUNT(*) FILTER (WHERE is_in_region IS NULL)  AS unknown,
                    COUNT(*)                                      AS total
                FROM accidents
                """
            )
        ).one()
        logger.info(
            "Distribution: in_region={} ({:.2%}), out_region={} ({:.2%}), "
            "unknown={} ({:.2%}), total={}",
            dist.in_region,
            dist.in_region / dist.total,
            dist.out_region,
            dist.out_region / dist.total,
            dist.unknown,
            dist.unknown / dist.total,
            dist.total,
        )
        return {
            "in_region": dist.in_region,
            "out_region": dist.out_region,
            "unknown": dist.unknown,
            "total": dist.total,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-name", default="primorye_krai_land")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Пересчитать is_in_region для всех строк (по умолчанию только NULL)",
    )
    args = parser.parse_args()

    update_is_in_region(region_name=args.region_name, force=args.force)


if __name__ == "__main__":
    main()
