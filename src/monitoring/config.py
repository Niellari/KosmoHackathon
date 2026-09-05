"""Настройки мониторинга отделены от конкурсного обучения."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "monitoring.yaml"


class MonitoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str = Field(min_length=1)
    history_years: int = Field(default=3, ge=1, le=5)
    max_area_ha: float = Field(default=5000, gt=0, le=10000)
    max_period_days: int = Field(default=366, ge=1, le=366)
    max_queued_jobs: int = Field(default=10, ge=1, le=100)
    scale_m: int = Field(default=20, ge=10, le=100)
    min_valid_fraction: float = Field(default=0.6, ge=0, le=1)
    min_pixel_count: int = Field(default=10, ge=1)
    max_interpolation_gap_days: int = Field(default=30, ge=1, le=90)
    database: Path
    cache_directory: Path


def load_monitoring_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if os.environ.get("EE_PROJECT_ID"):
        raw["project_id"] = os.environ["EE_PROJECT_ID"]
    config = MonitoringConfig.model_validate(raw)
    return config.model_copy(
        update={
            key: (path.parent / getattr(config, key)).resolve()
            for key in ("database", "cache_directory")
        }
    )
