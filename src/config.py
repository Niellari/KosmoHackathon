"""Строгая схема и загрузка YAML-конфигурации приложения."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    train_path: Path = Path("data/train_dataset.csv")
    test_path: Path = Path("data/test_dataset.csv")


class NeighborsConfig(StrictModel):
    enabled: bool = True
    count: int = Field(default=2, ge=1, le=2)
    include_distances: bool = True


class InterpolationConfig(StrictModel):
    baseline: bool = True
    linear: bool = True


class PolygonHistoryConfig(StrictModel):
    enabled: bool = True
    doy_window: int = Field(default=21, ge=1, le=183)
    weighting_scale: float = Field(default=7.0, gt=0)
    include_std: bool = True
    include_reference_years: bool = True


class CropCurveConfig(StrictModel):
    enabled: bool = True
    doy_window: int = Field(default=7, ge=1, le=183)
    weighting_scale: float = Field(default=3.0, gt=0)
    aggregation: Literal["median", "mean"] = "median"


class CalendarConfig(StrictModel):
    enabled: bool = True
    include_doy: bool = True
    cyclic_encoding: bool = True
    include_year: bool = True


class CropTypeConfig(StrictModel):
    enabled: bool = True
    encoding: Literal["one_hot"] = "one_hot"


class FeaturesConfig(StrictModel):
    neighbors: NeighborsConfig = NeighborsConfig()
    interpolation: InterpolationConfig = InterpolationConfig()
    polygon_history: PolygonHistoryConfig = PolygonHistoryConfig()
    crop_curve: CropCurveConfig = CropCurveConfig()
    calendar: CalendarConfig = CalendarConfig()
    crop_type: CropTypeConfig = CropTypeConfig()


class TrainingConfig(StrictModel):
    target_mode: Literal["direct", "residual"] = "direct"
    residual_baseline: Literal["neighbor_mean", "linear"] = "linear"


class ModelDefinition(StrictModel):
    type: Literal[
        "baseline", "heuristic_ensemble", "lightgbm", "catboost", "random_forest"
    ]
    artifact_path: Path | None = None
    load_if_exists: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(StrictModel):
    selected: str = "lightgbm"
    on_unavailable: Literal["error", "fallback"] = "error"
    fallback_to: str | None = None
    available: dict[str, ModelDefinition]

    @model_validator(mode="after")
    def selected_models_exist(self) -> "ModelsConfig":
        if self.selected not in self.available:
            raise ValueError(f"Модель {self.selected!r} отсутствует в models.available")
        if self.on_unavailable == "fallback":
            if not self.fallback_to or self.fallback_to not in self.available:
                raise ValueError("models.fallback_to должен указывать на доступную модель")
            if self.fallback_to == self.selected:
                raise ValueError("Fallback-модель должна отличаться от выбранной")
        return self


class PredictConfig(StrictModel):
    output_path: Path = Path("artifacts/submission.csv")
    prediction_column: str = "primary_ndvi_true"


class ValidationConfig(StrictModel):
    sample_size: int = Field(default=3000, ge=1)
    seed: int = 42


class ServerConfig(StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    auto_select_port: bool = True
    open_browser: bool = True
    debug: bool = False


class AppConfig(StrictModel):
    data: DataConfig = DataConfig()
    features: FeaturesConfig = FeaturesConfig()
    training: TrainingConfig = TrainingConfig()
    models: ModelsConfig
    predict: PredictConfig = PredictConfig()
    validation: ValidationConfig = ValidationConfig()
    server: ServerConfig = ServerConfig()

    @model_validator(mode="after")
    def residual_baseline_is_enabled(self) -> "AppConfig":
        if self.training.target_mode != "residual":
            return self
        baseline = self.training.residual_baseline
        if baseline == "linear" and not self.features.interpolation.linear:
            raise ValueError(
                "training.residual_baseline=linear требует "
                "features.interpolation.linear=true"
            )
        if baseline == "neighbor_mean" and not self.features.interpolation.baseline:
            raise ValueError(
                "training.residual_baseline=neighbor_mean требует "
                "features.interpolation.baseline=true"
            )
        return self


def load_config(path: Path | str = "config.yaml") -> AppConfig:
    """Читает YAML и возвращает проверенную конфигурацию."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфигурация не найдена: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Корень {config_path} должен быть YAML-объектом")
    config = AppConfig.model_validate(raw)

    # Пути из YAML считаются относительно самого файла конфигурации.
    base = config_path.resolve().parent
    data = config.data.model_copy(
        update={
            "train_path": _resolve_path(base, config.data.train_path),
            "test_path": _resolve_path(base, config.data.test_path),
        }
    )
    available = {
        name: definition.model_copy(
            update={
                "artifact_path": _resolve_path(base, definition.artifact_path)
                if definition.artifact_path is not None
                else None
            }
        )
        for name, definition in config.models.available.items()
    }
    models = config.models.model_copy(update={"available": available})
    predict = config.predict.model_copy(
        update={"output_path": _resolve_path(base, config.predict.output_path)}
    )
    return config.model_copy(
        update={"data": data, "models": models, "predict": predict}
    )


def _resolve_path(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else base / path


def select_model(config: AppConfig, name: str | None) -> AppConfig:
    """Возвращает копию конфига с CLI-переопределением выбранной модели."""

    if name is None:
        return config
    if name not in config.models.available:
        choices = ", ".join(sorted(config.models.available))
        raise ValueError(f"Неизвестная модель {name!r}. Доступны: {choices}")
    return config.model_copy(
        update={"models": config.models.model_copy(update={"selected": name})}
    )
