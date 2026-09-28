"""Input schema of the served model, and conversion of requests into model input.

Callers send real BAAC values by column name, for instance `"catv": "7"` for a
car. The model itself works on categories, and XGBoost stores the categories
it saw in training inside the model. Handing it a pandas categorical column
of raw values lets XGBoost re-code them to its own internal codes, so the
value-to-code mapping never has to be kept or shipped separately, and it
follows each retrained model automatically.

Everything here is read from the loaded model rather than hard-coded: a model
trained on different columns needs no code change.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Above this many allowed values, error messages and /model/schema give the
# count instead of the full list (the commune code alone has ~24k values).
MAX_LISTED_VALUES = 50


@dataclass
class FeatureSchema:
    """Column names in model order, and allowed values for categorical ones."""

    names: list[str]
    categories: dict[str, list[str]] = field(default_factory=dict)

    @property
    def numeric(self) -> list[str]:
        return [n for n in self.names if n not in self.categories]


class InvalidFeaturesError(ValueError):
    """Raised when a request does not match the model's schema."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def schema_from_model(model: Any) -> FeatureSchema:
    """Reads column names, types and categories from an XGBoost model."""
    booster = model.get_booster()
    names = list(booster.feature_names)
    exported = booster.get_categories(export_to_arrow=True).to_arrow()
    categories = {
        name: [str(v) for v in values.to_pylist()]
        for name, values in exported
        if values is not None
    }
    return FeatureSchema(names=names, categories=categories)


def _describe_allowed(values: list[str]) -> str:
    if len(values) <= MAX_LISTED_VALUES:
        return f"allowed: {', '.join(values)}"
    return f"{len(values)} allowed values, see GET /model/schema"


def _as_category_value(value: Any) -> str:
    """BAAC codes arrive as strings or numbers: compare them as strings."""
    if isinstance(value, bool):
        raise TypeError("booleans are not valid category values")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def to_model_input(features: dict[str, Any], schema: FeatureSchema) -> pd.DataFrame:
    """Validates a request against the schema and builds a one-row frame.

    Collects every problem before raising, so a caller fixes their request in
    one go rather than one field at a time. `None` means "unknown" and is
    passed on as a missing value, which XGBoost handles natively.
    """
    errors = []

    missing = [n for n in schema.names if n not in features]
    if missing:
        errors.append(f"missing features: {', '.join(missing)}")
    unexpected = sorted(set(features) - set(schema.names))
    if unexpected:
        errors.append(f"unknown features: {', '.join(unexpected)}")

    row: dict[str, Any] = {}
    for name in schema.names:
        if name not in features:
            continue
        value = features[name]

        if name in schema.categories:
            if value is None:
                row[name] = None
                continue
            try:
                code = _as_category_value(value)
            except TypeError as e:
                errors.append(f"{name}: {e}")
                continue
            allowed = schema.categories[name]
            if code not in allowed:
                errors.append(
                    f"{name}: value {code!r} was never seen in training "
                    f"({_describe_allowed(allowed)})"
                )
            row[name] = code
        else:
            if value is None:
                row[name] = np.nan
                continue
            try:
                row[name] = float(value)
            except (TypeError, ValueError):
                errors.append(f"{name}: expected a number, got {value!r}")

    if errors:
        raise InvalidFeaturesError(errors)

    frame = pd.DataFrame([row], columns=schema.names)
    for name, allowed in schema.categories.items():
        # Declare the model's own category list, not just the value received:
        # a lone missing value would otherwise leave the column with no
        # categories at all, which XGBoost cannot read.
        frame[name] = pd.Categorical(frame[name], categories=allowed)
    for name in schema.numeric:
        frame[name] = frame[name].astype(float)
    return frame


def describe(schema: FeatureSchema) -> list[dict[str, Any]]:
    """Human-readable schema, served on GET /model/schema."""
    out = []
    for name in schema.names:
        if name in schema.categories:
            values = schema.categories[name]
            entry: dict[str, Any] = {
                "name": name,
                "type": "categorical",
                "n_values": len(values),
            }
            if len(values) <= MAX_LISTED_VALUES:
                entry["values"] = values
        else:
            entry = {"name": name, "type": "numeric"}
        out.append(entry)
    return out
