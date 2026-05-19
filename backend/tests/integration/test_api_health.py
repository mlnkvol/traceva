def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_vlm_status_shape(client):
    response = client.get("/api/vlm/status")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"enabled", "model", "device", "ready", "load_failed"}
    assert isinstance(payload["enabled"], bool)
    assert isinstance(payload["ready"], bool)
    assert isinstance(payload["load_failed"], bool)
