import random
from locust import HttpUser, constant_pacing, task

class PredictRevenueUser(HttpUser):
    host = "http://localhost:8000"  # Ensure this matches your API's address
    wait_time = constant_pacing(1)  # 1 second between tasks

    @task
    def predict_revenue(self):
        # Generate valid random input data
        params = {
            "Gas_Price": round(random.uniform(1.0, 5.0), 2),
            "day_count": random.randint(1, 365),
            "Daily_Revenue_Lag14": round(random.uniform(1000.0, 10000.0), 2),
            "Daily_Revenue_Lag7": round(random.uniform(1000.0, 10000.0), 2),
            "CPI": round(random.uniform(1.0, 5.0), 2),
            "day_1": random.choice([0, 1]),
            "day_2": random.choice([0, 1]),
            "day_3": random.choice([0, 1]),
            "day_4": random.choice([0, 1]),
            "day_6": random.choice([0, 1]),
        }

        # Send the request as query parameters
        response = self.client.post("/predict_revenue", params=params)

        # Log responses for debugging
        print("Request data:", params)
        print("Response status:", response.status_code)
        print("Response body:", response.text)
