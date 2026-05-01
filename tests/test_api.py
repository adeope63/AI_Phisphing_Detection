import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_FILE = PROJECT_ROOT / 'frontend' / 'app.py'


@pytest.fixture
def client(monkeypatch):
    spec = importlib.util.spec_from_file_location('frontend_app', APP_FILE)
    frontend_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontend_app)

    def fake_predict(sender, subject, body, urls=''):
        if 'urgent' in body.lower():
            return 1, 0.91
        return 0, 0.72

    monkeypatch.setattr(frontend_app, 'predict_email_with_confidence', fake_predict)
    app = frontend_app.create_app()
    app.config['TESTING'] = True

    with app.test_client() as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'ok'


def test_predict_endpoint_success(client):
    response = client.post(
        '/predict',
        json={
            'sender': 'alerts@bank.com',
            'subject': 'Account warning',
            'body': 'Urgent action required',
            'urls': 'http://example.com'
        }
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['label'] == 'Phishing'
    assert payload['prediction'] == 1
    assert isinstance(payload['confidence'], float)


def test_predict_endpoint_validation_error(client):
    response = client.post('/predict', json={'sender': ''})
    assert response.status_code == 400
    payload = response.get_json()
    assert 'errors' in payload
    assert len(payload['errors']) >= 1
