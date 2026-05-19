from tests.conftest import wait_for_task


def test_prepare_returns_masks(client, synthetic_photo_image):
    with synthetic_photo_image.open("rb") as image_file:
        response = client.post(
            "/api/vectorize/prepare",
            files={"file": ("synthetic-photo.png", image_file, "image/png")},
            data={"mode": "semantic", "max_layers": "6"},
        )

    payload = response.json()
    assert response.status_code == 200
    assert isinstance(payload["task_id"], str)
    assert isinstance(payload["masks"], list)
    assert payload["masks"]


def test_finalize_after_prepare(client, synthetic_photo_image):
    with synthetic_photo_image.open("rb") as image_file:
        prepare = client.post(
            "/api/vectorize/prepare",
            files={"file": ("synthetic-photo.png", image_file, "image/png")},
            data={"mode": "semantic", "max_layers": "6"},
        )

    assert prepare.status_code == 200
    preview = prepare.json()
    layers = [
        {
            "id": mask["id"],
            "name": mask["name"],
            "color": mask["color"],
            "source_mask_ids": [mask["id"]],
        }
        for mask in preview["masks"]
    ]

    finalize = client.post(
        f"/api/vectorize/finalize/{preview['task_id']}",
        json={"tolerance": 1.0, "simplify": True, "layers": layers},
    )

    assert finalize.status_code == 200
    status = wait_for_task(client, preview["task_id"])
    assert status["status"] == "done"

    result = client.get(f"/api/result/{preview['task_id']}")
    payload = result.json()
    assert payload["svg_url"].startswith("/api/download/")
