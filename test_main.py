import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Assuming your FastAPI app is imported from your main file
from main import app

client = TestClient(app)

# Test /records/number_of_people_per_sensor/place/{url}/ endpoint
def test_get_number_of_people_per_sensor():
    response = client.get(
        "/records/number_of_people_per_sensor/place/dcuk/",
        params={"start": "2024-08-21T08:00:00", "stop": "2024-08-21T10:00:00"},
    )
    assert response.status_code == 200
    assert "sensors_estimates" in response.json()

def test_get_number_of_people_per_sensor_invalid_data():
    response = client.get(
        "/records/number_of_people_per_sensor/place/dcuk/",
        params={"start": "2024-08-21T08:00:00", "stop": "invalid"},
    )
    assert response.status_code == 422
    assert "detail" in response.json()

def test_get_number_of_people_per_sensor_no_data():
    response = client.get(
        "/records/number_of_people_per_sensor/place/unknown/",
        params={"start": "2024-08-21T08:00:00", "stop": "2024-08-21T10:00:00"},
    )
    assert response.status_code == 404
    assert "detail" in response.json()

# Test /records/number_of_people/place/{url}/ endpoint
def test_get_number_of_people_in_object():
    response = client.get(
        "/records/number_of_people/place/dcuk/",
        params={"start": "2024-08-21T08:00:00", "stop": "2024-08-21T10:00:00"},
    )
    assert response.status_code == 200
    assert "time_estimates" in response.json()

def test_get_number_of_people_in_object_invalid_data():
    response = client.get(
        "/records/number_of_people/place/dcuk/",
        params={"start": "2024-08-21T08:00:00", "stop": "invalid"},
    )
    assert response.status_code == 422
    assert "detail" in response.json()

def test_get_number_of_people_in_object_no_data():
    response = client.get(
        "/records/number_of_people/place/unknown/",
        params={"start": "2024-08-21T08:00:00", "stop": "2024-08-21T10:00:00"},
    )
    assert response.status_code == 404
    assert "detail" in response.json()

# Test /analyze_co2/ POST endpoint
def test_analyze_co2():
    response = client.post(
        "/analyze_co2/",
        params={
            "url": "dcuk",
            "start": "2024-08-14T08:00:00",
            "stop": "2024-08-14T10:00:00",
            "co2_threshold": 1000,
            "co2_rate_threshold": 10,
        },
    )
    assert response.status_code == 200
    assert "critical_devices" in response.json()

def test_analyze_co2_invalid_data():
    response = client.post(
        "/analyze_co2/",
        params={
            "url": "dcuk",
            "start": "2024-08-14T08:00:00",
            "stop": "invalid",
            "co2_threshold": 1000,
            "co2_rate_threshold": 10,
        },
    )
    assert response.status_code == 422
    assert "detail" in response.json()

def test_analyze_co2_no_critical_devices():
    response = client.post(
        "/analyze_co2/",
        params={
            "url": "unknown",
            "start": "2024-08-14T08:00:00",
            "stop": "2024-08-14T10:00:00",
            "co2_threshold": 1000,
            "co2_rate_threshold": 10,
        },
    )
    assert response.status_code == 404
    assert "detail" in response.json()

# Test /records/analyze_co2_dissipation/device/{device}/ endpoint
def test_analyze_co2_dissipation_device():
    response = client.get(
        "/records/analyze_co2_dissipation/device/eui-70b3d57ed006209f-co-05/",
        params={"start": "2024-08-14T08:00:00", "stop": "2024-08-14T10:00:00"},
    )
    assert response.status_code == 200
    assert "average_dissipation_time_minutes" in response.json()

def test_analyze_co2_dissipation_device_invalid_data():
    response = client.get(
        "/records/analyze_co2_dissipation/device/eui-70b3d57ed006209f-co-05/",
        params={"start": "2024-08-14T08:00:00", "stop": "invalid"},
    )
    assert response.status_code == 422
    assert "detail" in response.json()

def test_analyze_co2_dissipation_device_no_data():
    response = client.get(
        "/records/analyze_co2_dissipation/device/unknown/",
        params={"start": "2024-08-14T08:00:00", "stop": "2024-08-14T10:00:00"},
    )
    assert response.status_code == 404
    assert "detail" in response.json()

# Test /records/analyze_co2_dissipation/place/{url}/ endpoint
def test_analyze_place_co2_dissipation():
    response = client.get(
        "/records/analyze_co2_dissipation/place/dcuk/",
        params={"start": "2024-08-14T08:00:00", "stop": "2024-08-14T10:00:00"},
    )
    assert response.status_code == 200
    assert "average_dissipation_time_minutes" in response.json()

def test_analyze_place_co2_dissipation_invalid_data():
    response = client.get(
        "/records/analyze_co2_dissipation/place/dcuk/",
        params={"start": "2024-08-14T08:00:00", "stop": "invalid"},
    )
    assert response.status_code == 422
    assert "detail" in response.json()

def test_analyze_place_co2_dissipation_no_data():
    response = client.get(
        "/records/analyze_co2_dissipation/place/unknown/",
        params={"start": "2024-08-14T08:00:00", "stop": "2024-08-14T10:00:00"},
    )
    assert response.status_code == 404
    assert "detail" in response.json()

# Test /records/analyze_co2_changes/place/{url}/ endpoint
def test_analyze_place():
    response = client.get(
        "/records/analyze_co2_changes/place/dcuk/",
        params={"start": "2024-08-14T08:00:00", "stop": "2024-08-14T10:00:00"},
    )
    assert response.status_code == 200
    assert "co2_increase_count" in response.json()
    assert "co2_decrease_count" in response.json()

def test_analyze_place_invalid_data():
    response = client.get(
        "/records/analyze_co2_changes/place/dcuk/",
        params={"start": "2024-08-14T08:00:00", "stop": "invalid"},
    )
    assert response.status_code == 422
    assert "detail" in response.json()

def test_analyze_place_no_data():
    response = client.get(
        "/records/analyze_co2_changes/place/unknown/",
        params={"start": "2024-08-14T08:00:00", "stop": "2024-08-14T10:00:00"},
    )
    assert response.status_code == 404
    assert "detail" in response.json()

# Test /records/average_durations/{place} endpoint
def test_get_average_durations():
    response = client.get(
        "/records/average_durations/dcuk",
        params={"start": "2024-08-14T08:00:00", "stop": "2024-08-14T10:00:00"},
    )
    assert response.status_code == 200
    assert "average_ventilation_time" in response.json()
    assert "average_increase_time" in response.json()

def test_get_average_durations_invalid_data():
    response = client.get(
        "/records/average_durations/dcuk",
        params={"start": "2024-08-14T08:00:00", "stop": "invalid"},
    )
    assert response.status_code == 422
    assert "detail" in response.json()

def test_get_average_durations_no_data():
    response = client.get(
        "/records/average_durations/unknown",
        params={"start": "2024-08-14T08:00:00", "stop": "2024-08-14T10:00:00"},
    )
    assert response.status_code == 404
    assert "detail" in response.json()

# Test /records/number_of_zero_crossings/place/{device}/ endpoint
def test_get_number_of_zero_crossings():
    response = client.get(
        "/records/number_of_zero_crossings/place/eui-70b3d57ed006209f-co-05/",
        params={"start": "2024-08-14T08:00:00", "stop": "2024-08-14T10:00:00"},
    )
    assert response.status_code == 200
    assert "num_zero_crossings_down" in response.json()
    assert "num_zero_crossings_up" in response.json()

def test_get_number_of_zero_crossings_invalid_data():
    response = client.get(
        "/records/number_of_zero_crossings/place/eui-70b3d57ed006209f-co-05/",
        params={"start": "2024-08-14T08:00:00", "stop": "invalid"},
    )
    assert response.status_code == 422
    assert "detail" in response.json()

def test_get_number_of_zero_crossings_no_data():
    response = client.get(
        "/records/number_of_zero_crossings/place/unknown/",
        params={"start": "2024-08-14T08:00:00", "stop": "2024-08-14T10:00:00"},
    )
    assert response.status_code == 404
    assert "detail" in response.json()