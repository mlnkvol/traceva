from lxml import etree

from tests.conftest import wait_for_task


def test_vectorize_logo_mode_end_to_end(client, synthetic_logo_image):
    with synthetic_logo_image.open("rb") as image_file:
        response = client.post(
            "/api/vectorize",
            files={"file": ("synthetic-logo.png", image_file, "image/png")},
            data={"mode": "logo", "tolerance": "1.0", "max_layers": "4"},
        )

    assert response.status_code == 200
    task_id = response.json()["task_id"]
    status = wait_for_task(client, task_id)
    assert status["status"] == "done"

    result = client.get(f"/api/result/{task_id}")
    payload = result.json()
    assert payload["status"] == "done"
    assert payload["svg_url"].startswith("/api/download/")

    svg_response = client.get(payload["svg_url"])
    assert svg_response.status_code == 200
    assert "image/svg+xml" in svg_response.headers["content-type"]
    etree.fromstring(svg_response.content)


def test_vectorize_invalid_file(client):
    response = client.post(
        "/api/vectorize",
        files={"file": ("empty.txt", b"", "text/plain")},
        data={"mode": "logo"},
    )

    assert response.status_code in {400, 422}


def test_vectorize_status_endpoint(client, synthetic_logo_image):
    with synthetic_logo_image.open("rb") as image_file:
        response = client.post(
            "/api/vectorize",
            files={"file": ("status-logo.png", image_file, "image/png")},
            data={"mode": "logo", "tolerance": "1.0"},
        )

    task_id = response.json()["task_id"]
    status = wait_for_task(client, task_id)

    assert status["task_id"] == task_id
    assert status["status"] in {"done", "error"}
