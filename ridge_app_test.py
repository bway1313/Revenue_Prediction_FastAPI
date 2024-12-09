from fastapi.testclient import TestClient
from ridge_app import api 

client = TestClient(api)

def test_predict_revenue(mocker):
    # Arrange: Mock the predict function
    mocker.patch("ridge_app.predict", return_value=5000.0)

    # Prepare the parameters for the request
    params = {
        "Gas_Price": 3.5,
        "day_count": 10,
        "Daily_Revenue_Lag14": 1500.0,
        "Daily_Revenue_Lag7": 1600.0,
        "CPI": 2.8,
        "day_1": 0,
        "day_2": 1,
        "day_3": 0,
        "day_4": 0,
        "day_6": 0,
    }

    # Act: Send a POST request with the parameters
    response = client.post("/predict_revenue", params=params)

    # Assert: Verify the response status and body
    assert response.status_code == 200
    assert response.json() == {"predicted_revenue": 5000.0}

    
def test_predict_revenue_missing_params():
    # Act: Send a POST request with missing parameters
    params = {"Gas_Price": 3.5}  # Missing required fields
    response = client.post("/predict_revenue", params=params)

    # Assert: Expect a 422 error due to missing parameters
    assert response.status_code == 422
    assert "detail" in response.json()  # FastAPI includes validation details in the response
    
def test_predict_revenue_invalid_data():
    # Act: Send a POST request with invalid parameter types
    params = {
        "Gas_Price": "WRONG",  # Should be a float
        "day_count": 10,
        "Daily_Revenue_Lag14": 1500.0,
        "Daily_Revenue_Lag7": 1600.0,
        "CPI": 2.8,
        "day_1": 0,
        "day_2": 1,
        "day_3": 0,
        "day_4": 0,
        "day_6": 0,
    }
    response = client.post("/predict_revenue", params=params)

    # Assert: Expect a 422 error due to invalid data
    assert response.status_code == 422
    assert "detail" in response.json()
    
def test_predict_revenue_large_numbers(mocker):
    # Arrange: Mock the predict function
    mocker.patch("ridge_app.predict", return_value=1e9)

    # Prepare the input parameters
    params = {
        "Gas_Price": 1e6,
        "day_count": 1e5,
        "Daily_Revenue_Lag14": 1e8,
        "Daily_Revenue_Lag7": 1e8,
        "CPI": 10.0,
        "day_1": 1,
        "day_2": 1,
        "day_3": 1,
        "day_4": 1,
        "day_6": 1,
    }

    # Act: Send a POST request
    response = client.post("/predict_revenue", params=params)

    # Assert: Check the response
    assert response.status_code == 200
    assert response.json() == {"predicted_revenue": 1e9}
