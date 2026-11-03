from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

PAYLOAD = {
    "age": 35,
    "anciennete": 5,
    "revenu": 450000,
    "nb_reclamations": 2,
    "montant_mensuel": 25000,
    "retards_paiement": 2,
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "UP"


def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["version"]


def test_prediction():
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ("CHURN", "NO_CHURN")
    assert 0.0 <= data["probability"] <= 1.0
    assert "model_version" in data


def test_prediction_invalid_payload():
    # Champ manquant + valeur négative -> erreur de validation Pydantic (422)
    response = client.post("/predict", json={"age": -5})
    assert response.status_code == 422


def test_metrics():
    client.post("/predict", json=PAYLOAD)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "churn_predictions_total" in response.text
