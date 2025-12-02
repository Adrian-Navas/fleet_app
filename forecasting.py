import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ForecastModeler:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.numeric_features = ["rentals_lag_1", "rentals_lag_7"]
        self.categorical_features = ["day_of_week", "month", "is_weekend", "is_high_season"]
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
            ]
        )

    def train(self, X_train, y_train, model_type="Linear Regression", xgb_params=None):
        self.model_type = model_type
        
        if model_type == "Linear Regression":
            regressor = LinearRegression()
        elif model_type == "XGBoost":
            params = xgb_params if xgb_params else {}
            regressor = XGBRegressor(objective="reg:squarederror", random_state=42, **params)
        
        self.model = Pipeline(steps=[
            ("preprocess", self.preprocessor),
            ("model", regressor)
        ])
        
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)
        }
