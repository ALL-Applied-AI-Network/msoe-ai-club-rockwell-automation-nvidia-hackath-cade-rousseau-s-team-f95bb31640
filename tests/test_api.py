import pytest
from fastapi.testclient import TestClient

from api.main import DATA, app

client = TestClient(app)
needs_data = pytest.mark.skipif(
    not (DATA / "vial_fill" / "image_index.csv").exists(), reason="data/ not unzipped"
)


def test_image_rejects_other_folders():
    assert (
        client.get("/api/images/vial_fill/..%2F..%2Fpyproject.toml").status_code == 404
    )
    assert client.get("/api/images/secrets/x.jpg").status_code == 404


@needs_data
def test_recipes_and_frames():
    recipes = {r["recipe_id"]: r for r in client.get("/api/recipes").json()}
    assert sorted(recipes) == list(range(5, 15))
    assert (
        recipes[7]["ml"] == {"T100": 20, "T400": 50}
        and recipes[7]["clean_rgb"] is not None
    )

    frames = client.get("/api/videos/5/frames").json()
    assert len(frames) == 845
    assert (
        client.get(f"/api/images/vial_fill/{frames[0]['vial_fill']}").status_code == 200
    )
    assert all(f["in_fill"] is None for f in client.get("/api/videos/1/frames").json())


@needs_data
def test_inspection_and_schedule_report_pending_until_the_model_exists():
    result = client.get("/api/vials/7/inspection").json()
    assert (result["recipe_id"], result["prev_recipe_id"]) == (7, 6)
    assert client.get(result["image_url"]).headers["content-type"] == "image/jpeg"
    if result["source"] == "pending":
        assert result["status"] == "PENDING" and result["observed_rgb"] is None

    plan = client.post("/api/schedule", json={"recipe_ids": [6, 7, 13]}).json()
    assert [(t["prev_recipe_id"], t["recipe_id"]) for t in plan["transitions"]] == [
        (6, 7),
        (7, 13),
    ]
    assert client.post("/api/schedule", json={"recipe_ids": [6, 99]}).status_code == 422


@needs_data
def test_schedule_optimizes_with_fixed_first_batch_and_duplicate_recipes():
    submitted = [6, 7, 6, 12, 13]
    plan = client.post("/api/schedule", json={"recipe_ids": submitted}).json()

    assert plan["source"] == "model"
    assert plan["submitted"] == submitted
    assert plan["suggested"][0] == submitted[0]
    assert sorted(plan["suggested"]) == sorted(submitted)
    assert len(set(plan["suggested_batch_ids"])) == len(submitted)
    assert plan["suggested_cost"] <= plan["submitted_cost"]
    assert plan["suggested_worst_transition"] <= plan["submitted_worst_transition"]
    assert plan["method"] == "exact-held-karp-minimax"
    assert plan["warnings"]


@needs_data
def test_schedule_rejects_empty_queue():
    assert client.post("/api/schedule", json={"recipe_ids": []}).status_code == 422


@needs_data
def test_inspection_flags_the_carryover_vial_and_passes_its_batch():
    first, second, third = (client.get(f"/api/vials/{v}/inspection").json() for v in (7, 8, 9))
    assert first["source"] == "model" and first["status"] == "FAIL"
    assert "Recipe 6" in first["cause"]                     # names the recipe that ran before
    assert second["status"] == third["status"] == "PASS"
    assert second["cause"] is None and third["cause"] is None


def test_schedule_rejects_a_queue_too_long_for_exact_search():
    assert client.post("/api/schedule", json={"recipe_ids": [6] * 13}).status_code == 422
