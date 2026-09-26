"""Tests for the Prometheus metrics exposed on /metrics."""

from prometheus_client import REGISTRY

from tests.api_helpers import ADMIN_PW, FEATURES, OPERATOR_PW, SERVICE_PW, auth_header

# The `client` fixture lives in tests/conftest.py.


def sample(name: str, **labels) -> float:
    """Current value of a metric sample, 0 if it was never recorded."""
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_endpoint_is_public_and_in_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "api_model_loaded 1.0" in response.text


def test_served_model_version_is_exposed(client):
    assert sample("api_model_version_info", model_version="test") == 1.0


def test_prediction_is_counted_by_outcome_and_version(client):
    before = sample("api_predictions_total", outcome="severe", model_version="test")
    before_hist = sample("api_prediction_severe_probability_count")

    response = client.post(
        "/predict", json=FEATURES, headers=auth_header(client, "operator1", OPERATOR_PW)
    )
    assert response.status_code == 200

    # The stub model returns 0.8 for the severe class, above the threshold.
    after = sample("api_predictions_total", outcome="severe", model_version="test")
    assert after == before + 1
    assert sample("api_prediction_severe_probability_count") == before_hist + 1


def test_failed_login_is_counted(client):
    before = sample("api_login_attempts_total", result="failure")
    client.post("/token", data={"username": "operator1", "password": "wrong-password"})
    assert sample("api_login_attempts_total", result="failure") == before + 1


def test_requests_are_labelled_by_route_template(client):
    """A username in the path must not become its own time series."""
    headers = auth_header(client, "admin1", ADMIN_PW)
    before = sample(
        "api_http_requests_total",
        method="DELETE",
        route="/admin/users/{username}",
        status="404",
    )

    client.delete("/admin/users/ghost", headers=headers)

    after = sample(
        "api_http_requests_total",
        method="DELETE",
        route="/admin/users/{username}",
        status="404",
    )
    assert after == before + 1
    assert (
        sample(
            "api_http_requests_total",
            method="DELETE",
            route="/admin/users/ghost",
            status="404",
        )
        == 0.0
    )


def test_unknown_paths_share_one_label(client):
    before = sample(
        "api_http_requests_total", method="GET", route="unmatched", status="404"
    )
    client.get("/does-not-exist")
    client.get("/another-random-path")
    after = sample(
        "api_http_requests_total", method="GET", route="unmatched", status="404"
    )
    assert after == before + 2


def test_scraping_metrics_is_not_counted_as_traffic(client):
    before = sample(
        "api_http_requests_total", method="GET", route="/metrics", status="200"
    )
    client.get("/metrics")
    assert (
        sample("api_http_requests_total", method="GET", route="/metrics", status="200")
        == before
    )


def test_model_reload_is_counted(client):
    before = sample("api_model_reloads_total", result="success")
    response = client.post(
        "/admin/reload-model", headers=auth_header(client, "airflow1", SERVICE_PW)
    )
    assert response.status_code == 200
    assert sample("api_model_reloads_total", result="success") == before + 1
