"""Tests for the release gate in src/training/promote.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import promote  # noqa: E402

# Metrics of a healthy model, measured on test.parquet at threshold 0.35.
GOOD_METRICS = {"auc_roc": 0.8763, "recall": 0.8961, "precision": 0.3800}

# A model that still looks plausible but lost a third of its recall.
DEGRADED_METRICS = {"auc_roc": 0.8388, "recall": 0.6537, "precision": 0.4728}


def test_good_metrics_pass():
    passed, reasons = promote.meets_thresholds(GOOD_METRICS)
    assert passed
    assert reasons == []


def test_degraded_model_is_refused():
    """The main case the gate exists for: plausible metrics, ruined recall."""
    passed, reasons = promote.meets_thresholds(DEGRADED_METRICS)
    assert not passed
    assert len(reasons) == 2  # both auc and recall below threshold


def test_low_auc_is_refused():
    passed, reasons = promote.meets_thresholds({**GOOD_METRICS, "auc_roc": 0.5})
    assert not passed
    assert any("auc_roc" in r for r in reasons)


def test_low_recall_is_refused():
    """Recall matters most: a severe accident reported as minor is the worst case."""
    passed, reasons = promote.meets_thresholds({**GOOD_METRICS, "recall": 0.1})
    assert not passed
    assert any("recall" in r for r in reasons)


def test_missing_metrics_are_refused():
    """No metrics means evaluate.py never ran: the model is unmeasured."""
    passed, reasons = promote.meets_thresholds({})
    assert not passed
    assert len(reasons) == 2


def test_empty_metrics_do_not_promote_silently():
    """A metric at 0.0 must be treated as a real value, not as missing."""
    passed, reasons = promote.meets_thresholds({"auc_roc": 0.0, "recall": 0.0})
    assert not passed
    assert all("missing" not in r for r in reasons)


def test_thresholds_are_configurable(monkeypatch):
    monkeypatch.setattr(promote, "MIN_AUC_ROC", 0.99)
    passed, reasons = promote.meets_thresholds(GOOD_METRICS)
    assert not passed
    assert any("0.99" in r for r in reasons)


def test_value_exactly_at_threshold_passes():
    metrics = {"auc_roc": promote.MIN_AUC_ROC, "recall": promote.MIN_RECALL_SEVERE}
    passed, _ = promote.meets_thresholds(metrics)
    assert passed


class FakeRegistry:
    """Records the alias and tags promote() sets, like MlflowClient would."""

    def __init__(self, production=None):
        self.alias = production
        self.tags = {production: "production"} if production else {}

    def get_model_version_by_alias(self, name, alias):
        if self.alias is None:
            raise RuntimeError("no alias")
        return type("Version", (), {"version": self.alias})()

    def set_registered_model_alias(self, name, alias, version):
        self.alias = version

    def set_model_version_tag(self, name, version, key, value):
        self.tags[version] = value

    def delete_model_version_tag(self, name, version, key):
        self.tags.pop(version, None)


def test_stage_tag_moves_with_the_alias():
    registry = FakeRegistry(production="7")
    promote.promote(registry, "8")
    assert registry.alias == "8"
    assert registry.tags == {"8": "production"}


def test_first_promotion_without_previous_version():
    registry = FakeRegistry()
    promote.promote(registry, "1")
    assert registry.alias == "1"
    assert registry.tags == {"1": "production"}
