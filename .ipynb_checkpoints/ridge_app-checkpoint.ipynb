{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "29c81f4a-ceba-4480-9554-fac6e4481493",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from fastapi import FastAPI\n",
    "import joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from pydantic import BaseModel, PositiveFloat, Field"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "5326fc8b-4b69-4ace-87e8-a37c1d6bcc83",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "ML_MODEL = joblib.load(\"./ridge_best_model.joblib\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "95223008-0c25-4cdb-838e-59dd5432c566",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# FastAPI.\n",
    "api_title = \"RevenueApp\"\n",
    "api_description = \"\"\"RevenueApp allows you to predict the grocery revenue on a given date\"\"\"\n",
    "api = FastAPI(title=api_title, description=api_description)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8edefa77-2f4d-4ead-b583-9417683b8991",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "class Gas_Price(BaseModel):\n",
    "    \"\"\"\n",
    "    Data model for gas price.\n",
    "    \"\"\"\n",
    "    Gas_Price: PositiveFloat\n",
    "    \n",
    "class day_count(BaseModel):\n",
    "    \"\"\"\n",
    "    Data model for day's since Dec 19, 2010.\n",
    "    \"\"\"\n",
    "    day_count: PositiveFloat\n",
    "\n",
    "class Daily_Revenue_Lag14(BaseModel):\n",
    "    \"\"\"\n",
    "    Data model for revenue lagged 14 business days.\n",
    "    \"\"\"\n",
    "    Daily_Revenue_Lag14: PositiveFloat\n",
    "    \n",
    "class Daily_Revenue_Lag7(BaseModel):\n",
    "    \"\"\"\n",
    "    Data model for revenue lagged 7 business days.\n",
    "    \"\"\"\n",
    "    Daily_Revenue_Lag7: PositiveFloat\n",
    "    \n",
    "class CPI(BaseModel):\n",
    "    \"\"\"\n",
    "    Data model for Consumer Price Index (CPI).\n",
    "    \"\"\"\n",
    "    CPI: PositiveFloat\n",
    "    \n",
    "class day_1(BaseModel):\n",
    "    day_1: int = Field(..., ge=0, le=1)  # Only allow 0 or 1\n",
    "    \n",
    "class day_2(BaseModel):\n",
    "    day_2: int = Field(..., ge=0, le=1)  # Only allow 0 or 1    \n",
    "\n",
    "class day_3(BaseModel):\n",
    "    day_3: int = Field(..., ge=0, le=1)  # Only allow 0 or 1    \n",
    "    \n",
    "class day_4(BaseModel):\n",
    "    day_4: int = Field(..., ge=0, le=1)  # Only allow 0 or 1\n",
    "    \n",
    "class day_6(BaseModel):\n",
    "    day_6: int = Field(..., ge=0, le=1)  # Only allow 0 or 1\n",
    "    \n",
    "    \n",
    "class predicted_revenue(BaseModel):\n",
    "    \"\"\"\n",
    "    Data model for revenue.\n",
    "    \"\"\"\n",
    "    # Our simple linear regression model does not make only positive\n",
    "    # predictions. So, technically, we can only guarantee that we will return\n",
    "    # a float. We can't guarantee it will be a PositiveFloat.\n",
    "    predicted_revenue: float\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4b8f8bea-8c3a-4986-adf4-2cfefe2b9bb4",
   "metadata": {},
   "outputs": [],
   "source": [
    "FEATURE_NAMES = ['Gas_Price', 'CPI', 'Daily_Revenue_Lag14', 'Daily_Revenue_Lag7', 'day_count','day_1', 'day_2', 'day_3', 'day_4', 'day_6']\n",
    "\n",
    "def predict(\n",
    "    Gas_Price: float,\n",
    "    day_count: float,\n",
    "    Daily_Revenue_Lag14: float,\n",
    "    Daily_Revenue_Lag7: float,\n",
    "    CPI: float,\n",
    "    day_1: int,\n",
    "    day_2: int,\n",
    "    day_3: int,\n",
    "    day_4: int,\n",
    "    day_6: int,\n",
    ") -> float:\n",
    "    \"\"\"\n",
    "    Utility to make predictions from the ML model.\n",
    "    \"\"\"\n",
    "    # Ensure all inputs are numeric\n",
    "    try:\n",
    "        Gas_Price = float(Gas_Price)\n",
    "        day_count = float(day_count)\n",
    "        Daily_Revenue_Lag14 = float(Daily_Revenue_Lag14)\n",
    "        Daily_Revenue_Lag7 = float(Daily_Revenue_Lag7)\n",
    "        CPI = float(CPI)\n",
    "        day_1 = int(day_1)\n",
    "        day_2 = int(day_2)\n",
    "        day_3 = int(day_3)\n",
    "        day_4 = int(day_4)\n",
    "        day_6 = int(day_6)\n",
    "    except ValueError as e:\n",
    "        raise ValueError(f\"Invalid input data: {e}\")\n",
    "\n",
    "    # Create a DataFrame with valid feature names\n",
    "    model_input = pd.DataFrame(\n",
    "        [[Gas_Price, CPI, Daily_Revenue_Lag14, Daily_Revenue_Lag7, day_count,day_1, day_2, day_3, day_4, day_6]],\n",
    "        columns=FEATURE_NAMES\n",
    "    )\n",
    "\n",
    "    # Use the ML model to predict revenue\n",
    "    return ML_MODEL.predict(model_input)[0]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5662dbd3-bd5f-4818-a6e9-3038145c4744",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "@api.post(\"/predict_revenue\")\n",
    "def predict_revenue(\n",
    "    Gas_Price: float,\n",
    "    day_count: float,\n",
    "    Daily_Revenue_Lag14: float,\n",
    "    Daily_Revenue_Lag7: float,\n",
    "    CPI: float,\n",
    "    day_1: int,\n",
    "    day_2: int,\n",
    "    day_3: int,\n",
    "    day_4: int,\n",
    "    day_6: int,\n",
    "):\n",
    "    \"\"\"\n",
    "    Endpoint to predict revenue based on input features.\n",
    "    \"\"\"\n",
    "    try:\n",
    "        predicted_revenue = predict(Gas_Price, day_count, Daily_Revenue_Lag14, Daily_Revenue_Lag7, CPI, day_1, day_2, day_3, day_4, day_6)\n",
    "        return {\"predicted_revenue\": predicted_revenue}\n",
    "    except ValueError as e:\n",
    "        return {\"error\": str(e)}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "5382be14-32b7-450c-b065-96bf528ba5fd",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Features seen during training: ['Gas_Price' 'CPI' 'Daily_Revenue_Lag14' 'Daily_Revenue_Lag7' 'day_count'\n",
      " 'day_1' 'day_2' 'day_3' 'day_4' 'day_6']\n"
     ]
    }
   ],
   "source": [
    "print(\"Features seen during training:\", ML_MODEL.feature_names_in_)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "797aae15-f54d-4904-8517-6d2b0bd2ea47",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
