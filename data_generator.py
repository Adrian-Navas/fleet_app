import pandas as pd
import numpy as np
from datetime import timedelta

def generate_stations(capacity_factor: float = 1.0):
    """
    Generates a list of 50+ stations in Spain.

    Args:
        capacity_factor: Multiplicative factor applied to every station capacity.
    """
    # Simplified list of major cities/locations
    # Capacity reduced by 20% as per request to increase utilization pressure
    stations_data = [
        {"city": "Madrid", "lat": 40.4168, "lon": -3.7038, "base_demand": 100, "capacity": 128, "segment": "City"},
        {"city": "Madrid Airport", "lat": 40.4983, "lon": -3.5676, "base_demand": 150, "capacity": 192, "segment": "Airport"},
        {"city": "Barcelona", "lat": 41.3851, "lon": 2.1734, "base_demand": 90, "capacity": 115, "segment": "City"},
        {"city": "Barcelona Airport", "lat": 41.2974, "lon": 2.0833, "base_demand": 160, "capacity": 224, "segment": "Airport"},
        {"city": "Valencia", "lat": 39.4699, "lon": -0.3763, "base_demand": 60, "capacity": 77, "segment": "City"},
        {"city": "Sevilla", "lat": 37.3891, "lon": -5.9845, "base_demand": 55, "capacity": 70, "segment": "City"},
        {"city": "Málaga", "lat": 36.7213, "lon": -4.4214, "base_demand": 70, "capacity": 90, "segment": "City"},
        {"city": "Málaga Airport", "lat": 36.6749, "lon": -4.4991, "base_demand": 130, "capacity": 192, "segment": "Airport"},
        {"city": "Bilbao", "lat": 43.2630, "lon": -2.9350, "base_demand": 40, "capacity": 51, "segment": "City"},
        {"city": "Palma de Mallorca", "lat": 39.5696, "lon": 2.6502, "base_demand": 80, "capacity": 128, "segment": "City"},
        {"city": "Palma Airport", "lat": 39.5517, "lon": 2.7388, "base_demand": 180, "capacity": 256, "segment": "Airport"},
        {"city": "Alicante", "lat": 38.3452, "lon": -0.4810, "base_demand": 50, "capacity": 64, "segment": "City"},
        {"city": "Granada", "lat": 37.1773, "lon": -3.5986, "base_demand": 35, "capacity": 45, "segment": "City"},
        {"city": "Zaragoza", "lat": 41.6488, "lon": -0.8891, "base_demand": 30, "capacity": 38, "segment": "City"},
        {"city": "Ibiza", "lat": 38.9067, "lon": 1.4206, "base_demand": 60, "capacity": 96, "segment": "City"}, # High seasonality
        {"city": "Tenerife Sur", "lat": 28.05, "lon": -16.57, "base_demand": 90, "capacity": 128, "segment": "Airport"},
        # Additional cities for denser coverage
        {"city": "Santander", "lat": 43.4623, "lon": -3.8100, "base_demand": 25, "capacity": 32, "segment": "City"},
        {"city": "A Coruña", "lat": 43.3713, "lon": -8.3960, "base_demand": 30, "capacity": 38, "segment": "City"},
        {"city": "Vigo", "lat": 42.2406, "lon": -8.7207, "base_demand": 28, "capacity": 35, "segment": "City"},
        {"city": "Murcia", "lat": 37.9922, "lon": -1.1307, "base_demand": 35, "capacity": 44, "segment": "City"},
        {"city": "Córdoba", "lat": 37.8882, "lon": -4.7794, "base_demand": 30, "capacity": 38, "segment": "City"},
        {"city": "Valladolid", "lat": 41.6528, "lon": -4.7245, "base_demand": 32, "capacity": 42, "segment": "City"},
        {"city": "San Sebastián", "lat": 43.3183, "lon": -1.9812, "base_demand": 38, "capacity": 48, "segment": "City"},
        {"city": "Tarragona", "lat": 41.1189, "lon": 1.2445, "base_demand": 22, "capacity": 29, "segment": "City"},
        {"city": "Girona", "lat": 41.9794, "lon": 2.8214, "base_demand": 20, "capacity": 26, "segment": "City"},
        {"city": "Las Palmas", "lat": 28.1248, "lon": -15.4300, "base_demand": 65, "capacity": 96, "segment": "City"},
    ]
    
    # Expand to ~50 stations
    expanded_stations = []
    count = 1
    for s in stations_data:
        scaled_capacity = int(round(s["capacity"] * capacity_factor))

        expanded_stations.append({
            "station_id": count,
            "station_name": f"{s['city']} Central",
            **s
        })
        expanded_stations[-1]["capacity"] = scaled_capacity
        count += 1
        # Add secondary stations for big cities
        if s["base_demand"] > 80 and s["segment"] == "City":
            secondary_capacity = int(round(scaled_capacity * 0.6))
            expanded_stations.append({
                "station_id": count,
                "station_name": f"{s['city']} North",
                "city": s["city"],
                "lat": s["lat"] + 0.02,
                "lon": s["lon"] + 0.02,
                "base_demand": s["base_demand"] * 0.6,
                "capacity": secondary_capacity,
                "segment": "City"
            })
            count += 1
            
    return pd.DataFrame(expanded_stations)

def generate_daily_demand(stations_df):
    """
    Generates daily demand for each station for the last 2 years.
    """
    end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.DateOffset(years=2)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    all_data = []
    
    np.random.seed(42)
    
    # Common holidays in Spain (fixed dates for simplicity)
    holidays = [
        (1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)
    ]
    
    for _, station in stations_df.iterrows():
        base = station["base_demand"]
        
        for date in dates:
            # Seasonality Factors
            month = date.month
            dow = date.dayofweek
            day = date.day
            
            is_holiday = (month, day) in holidays
            
            season_factor = 1.0
            # Summer peak (Stronger influence)
            if month in [7, 8]:
                season_factor = 1.8 if station["segment"] == "Airport" else 1.4
            # Shoulder season
            elif month in [6, 9]:
                season_factor = 1.3
            # Winter dip
            elif month in [1, 2, 11]:
                season_factor = 0.8
                
            # Weekend factor (Stronger influence)
            weekend_factor = 1.0
            if dow >= 5: # Sat, Sun
                weekend_factor = 1.6 if station["segment"] == "Airport" else 0.6 # Airports very busy, City very quiet
            elif dow == 0 or dow == 4: # Mon, Fri
                weekend_factor = 1.3 # Business travel peak
            
            # Holiday factor
            holiday_factor = 1.0
            if is_holiday:
                holiday_factor = 1.5 if station["segment"] == "Airport" else 0.5 # Holidays: Airports busy, City quiet
                
            # Specific Island seasonality (Ibiza, Palma)
            if "Palma" in station["city"] or "Ibiza" in station["city"]:
                if month in [6, 7, 8, 9]:
                    season_factor *= 1.6 # Even stronger summer peak
            
            # Noise (Reduced significantly)
            # Using a consistent noise pattern based on date to make it "predictable" but slightly variable
            # or just very low random noise
            noise = np.random.normal(0, 0.1)
            
            # Final Demand (Deterministic, no Poisson)
            # Base * Season * Weekend * Holiday * Noise
            lambda_val = base * season_factor * weekend_factor * holiday_factor * (1 + noise)
            
            # Ensure non-negative and integer
            rentals = int(round(max(0, lambda_val)))
            
            all_data.append({
                "station_id": station["station_id"],
                "date": date,
                "rentals": rentals,
                "month": month,
                "day_of_week": dow,
                "is_weekend": 1 if dow >= 5 else 0,
                "is_high_season": 1 if month in [6, 7, 8, 9] else 0
            })
            
    df = pd.DataFrame(all_data)
    
    # Add Lags
    df = df.sort_values(["station_id", "date"])
    df["rentals_lag_1"] = df.groupby("station_id")["rentals"].shift(1)
    df["rentals_lag_7"] = df.groupby("station_id")["rentals"].shift(7)
    
    # Drop NaN from lags
    df = df.dropna()
    
    return df

def generate_fleet_state(stations_df, date):
    """
    Generates a snapshot of the fleet for a specific date.
    """
    # Randomly assign cars based on capacity, but with some imbalance to create optimization need
    state = []
    for _, row in stations_df.iterrows():
        # Random utilization between 20% and 90%
        util = np.random.uniform(0.2, 0.9)
        cars = int(row["capacity"] * util)
        state.append({
            "station_id": row["station_id"],
            "date": date,
            "cars_available": cars,
            "capacity": row["capacity"]
        })
    return pd.DataFrame(state)
