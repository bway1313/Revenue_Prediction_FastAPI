from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, PositiveFloat, Field

ML_MODEL = joblib.load("./ridge_best_model.joblib")

# FastAPI.
api_title = "RevenueApp"
api_description = """RevenueApp allows you to predict the grocery revenue on a given date"""
api = FastAPI(title=api_title, description=api_description)

class Gas_Price(BaseModel):
    """
    Data model for gas price.
    """
    Gas_Price: PositiveFloat
    
class day_count(BaseModel):
    """
    Data model for day's since Dec 19, 2010.
    """
    day_count: PositiveFloat

class Daily_Revenue_Lag14(BaseModel):
    """
    Data model for revenue lagged 14 business days.
    """
    Daily_Revenue_Lag14: PositiveFloat
    
class Daily_Revenue_Lag7(BaseModel):
    """
    Data model for revenue lagged 7 business days.
    """
    Daily_Revenue_Lag7: PositiveFloat
    
class CPI(BaseModel):
    """
    Data model for Consumer Price Index (CPI).
    """
    CPI: PositiveFloat
    
class day_1(BaseModel):
    day_1: int = Field(..., ge=0, le=1)  # Only allow 0 or 1
    
class day_2(BaseModel):
    day_2: int = Field(..., ge=0, le=1)  # Only allow 0 or 1    

class day_3(BaseModel):
    day_3: int = Field(..., ge=0, le=1)  # Only allow 0 or 1    
    
class day_4(BaseModel):
    day_4: int = Field(..., ge=0, le=1)  # Only allow 0 or 1
    
class day_6(BaseModel):
    day_6: int = Field(..., ge=0, le=1)  # Only allow 0 or 1
    
    
class predicted_revenue(BaseModel):
    """
    Data model for revenue.
    """
    # Our simple linear regression model does not make only positive
    # predictions. So, technically, we can only guarantee that we will return
    # a float. We can't guarantee it will be a PositiveFloat.
    predicted_revenue: float

FEATURE_NAMES = ['Gas_Price', 'CPI', 'Daily_Revenue_Lag14', 'Daily_Revenue_Lag7', 'day_count','day_1', 'day_2', 'day_3', 'day_4', 'day_6']

def predict(
    Gas_Price: float,
    day_count: float,
    Daily_Revenue_Lag14: float,
    Daily_Revenue_Lag7: float,
    CPI: float,
    day_1: int,
    day_2: int,
    day_3: int,
    day_4: int,
    day_6: int,
) -> float:
    """
    Utility to make predictions from the ML model.
    """
    # Ensure all inputs are numeric
    try:
        Gas_Price = float(Gas_Price)
        day_count = float(day_count)
        Daily_Revenue_Lag14 = float(Daily_Revenue_Lag14)
        Daily_Revenue_Lag7 = float(Daily_Revenue_Lag7)
        CPI = float(CPI)
        day_1 = int(day_1)
        day_2 = int(day_2)
        day_3 = int(day_3)
        day_4 = int(day_4)
        day_6 = int(day_6)
    except ValueError as e:
        raise ValueError(f"Invalid input data: {e}")

    # Create a DataFrame with valid feature names
    model_input = pd.DataFrame(
        [[Gas_Price, CPI, Daily_Revenue_Lag14, Daily_Revenue_Lag7, day_count,day_1, day_2, day_3, day_4, day_6]],
        columns=FEATURE_NAMES
    )

    # Use the ML model to predict revenue
    return ML_MODEL.predict(model_input)[0]


@api.post("/predict_revenue")
def predict_revenue(
    Gas_Price: float,
    day_count: float,
    Daily_Revenue_Lag14: float,
    Daily_Revenue_Lag7: float,
    CPI: float,
    day_1: int,
    day_2: int,
    day_3: int,
    day_4: int,
    day_6: int,
):
    """
    Endpoint to predict revenue based on input features.
    """
    try:
        predicted_revenue = predict(Gas_Price, day_count, Daily_Revenue_Lag14, Daily_Revenue_Lag7, CPI, day_1, day_2, day_3, day_4, day_6)
        return {"predicted_revenue": predicted_revenue}
    except ValueError as e:
        return {"error": str(e)}
