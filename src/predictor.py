"""Жизненный цикл выбранной модели: fallback, кэширование и инференс."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile

# LightGBM может импортировать matplotlib при распаковке модели.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agropulse-matplotlib")
)
import joblib
import pandas as pd

from src.config import FeaturesConfig, ModelsConfig, TrainingConfig
from src.features import FeatureBuilder
from src.models import create_model
from src.models.base import GapModel, ModelUnavailableError


class PredictorService:
    def __init__(
        self,
        models: ModelsConfig,
        features: FeaturesConfig,
        training: TrainingConfig,
    ):
        self.models_config = models
        self.feature_builder = FeatureBuilder(features)
        self.training_config = training
        self.selected_name = models.selected
        self.definition = models.available[self.selected_name]
        self.model: GapModel = create_model(
            self.selected_name, self.definition, self.training_config
        )
        self._prepared = False

    def _config_hash(self) -> str:
        payload = {
            "features": self.feature_builder.config.model_dump(mode="json"),
            "training": self.training_config.model_dump(mode="json"),
            "type": self.definition.type,
            "params": self.definition.params,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _data_signature(frame: pd.DataFrame) -> dict:
        known = pd.to_numeric(frame["primary_ndvi"], errors="coerce")
        return {
            "rows": len(frame),
            "known_targets": int(known.notna().sum()),
            "target_sum": round(float(known.sum()), 8),
            "date_min": str(frame["date"].min()),
            "date_max": str(frame["date"].max()),
        }

    def _metadata_path(self, artifact: Path) -> Path:
        return artifact.with_suffix(artifact.suffix + ".metadata.json")

    def _load_artifact(self, train: pd.DataFrame) -> bool:
        artifact = self.definition.artifact_path
        if not self.definition.load_if_exists or artifact is None or not artifact.exists():
            return False
        metadata_path = self._metadata_path(artifact)
        if not metadata_path.exists():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("config_hash") != self._config_hash():
                return False
            if metadata.get("data_signature") != self._data_signature(train):
                return False
            loaded = joblib.load(artifact)
        except (
            OSError,
            ValueError,
            TypeError,
            ImportError,
            EOFError,
            pickle.UnpicklingError,
            json.JSONDecodeError,
        ):
            return False
        if not isinstance(loaded, GapModel):
            return False
        self.model = loaded
        print(f"Модель загружена: {artifact}")
        return True

    def _save_artifact(self, train: pd.DataFrame) -> None:
        artifact = self.definition.artifact_path
        if artifact is None:
            return
        artifact.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, artifact)
        metadata = {
            "model_name": self.selected_name,
            "model_type": self.definition.type,
            "training": self.training_config.model_dump(mode="json"),
            "feature_names": self.feature_builder.feature_names,
            "config_hash": self._config_hash(),
            "data_signature": self._data_signature(train),
        }
        self._metadata_path(artifact).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _activate_fallback(self) -> None:
        fallback = self.models_config.fallback_to
        if self.models_config.on_unavailable != "fallback" or fallback is None:
            raise RuntimeError("Fallback-модель не настроена")
        self.selected_name = fallback
        self.definition = self.models_config.available[fallback]
        self.model = create_model(
            fallback, self.definition, self.training_config
        )
        print(f"Используется fallback-модель: {fallback}")

    def prepare(self, train: pd.DataFrame) -> "PredictorService":
        if self._prepared:
            return self
        if self._load_artifact(train):
            self._prepared = True
            return self
        try:
            self.model.fit(train, self.feature_builder)
        except ModelUnavailableError as error:
            if self.models_config.on_unavailable != "fallback":
                raise
            print(f"{error}")
            self._activate_fallback()
            self.model.fit(train, self.feature_builder)
        self._save_artifact(train)
        self._prepared = True
        return self

    def predict(self, context: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        if not self._prepared:
            raise RuntimeError("PredictorService необходимо подготовить перед predict")
        return self.model.predict(context, targets, self.feature_builder)
