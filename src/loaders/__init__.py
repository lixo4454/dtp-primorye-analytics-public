"""Загрузчики данных из внешних источников."""

from src.loaders.dtp_stat_accumulated_downloader import (
    download_archive,
    extract_archive,
    extract_primorye,
    prepare_data,
)
from src.loaders.dtp_stat_accumulated_loader import load_to_db
from src.loaders.dtp_stat_accumulated_parser import parse_record

__all__ = [
    "download_archive",
    "extract_archive",
    "extract_primorye",
    "prepare_data",
    "parse_record",
    "load_to_db",
]
