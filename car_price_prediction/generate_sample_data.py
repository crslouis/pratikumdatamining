"""
Generate a synthetic sample CSV mimicking the schema + realistic correlations
of the Kaggle "Automotive Price Prediction Dataset"
(metawave/vehicle-price-prediction), for local testing before the real
dataset (1,000,000 rows) is downloaded.

Documented generation logic of the real dataset:
- Depreciation: vehicle age is the primary price factor (exponential decay)
- Mileage: correlated with age, negatively impacts price
- Horsepower: higher HP -> higher price
- Brand: each brand has a different base price

Download the real dataset from:
https://www.kaggle.com/datasets/metawave/vehicle-price-prediction
and replace sample_vehicle_price.csv before final submission.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 8000

brands = ["Toyota", "Ford", "Honda", "Chevrolet", "Nissan", "BMW", "Hyundai", "Kia"]
brand_base_price = {
    "Toyota": 24000, "Ford": 26000, "Honda": 23000, "Chevrolet": 25000,
    "Nissan": 22000, "BMW": 42000, "Hyundai": 21000, "Kia": 20000,
}
fuel_types = ["Gasoline", "Diesel", "Electric", "Hybrid"]
transmissions = ["Automatic", "Manual"]
drive_types = ["FWD", "RWD", "AWD"]
body_styles = ["Sedan", "SUV", "Pickup Truck", "Hatchback"]

rows = []
current_year = 2026
for _ in range(N):
    brand = np.random.choice(brands)
    year = np.random.randint(2005, 2026)
    age = current_year - year
    mileage = max(0, int(age * np.random.normal(12000, 3000) + np.random.normal(0, 5000)))
    horsepower = int(np.random.normal(200, 60))
    horsepower = max(80, horsepower)

    base = brand_base_price[brand]
    depreciation = base * (0.90 ** age)  # exponential decay with age
    mileage_penalty = mileage * 0.03
    hp_bonus = horsepower * 25
    noise = np.random.normal(0, 800)

    price = max(2000, depreciation - mileage_penalty + hp_bonus + noise)

    rows.append({
        "brand": brand,
        "model": f"{brand}_Model_{np.random.randint(1,6)}",
        "year": year,
        "mileage": mileage,
        "horsepower": horsepower,
        "transmission": np.random.choice(transmissions),
        "fuel_type": np.random.choice(fuel_types, p=[0.55, 0.2, 0.15, 0.1]),
        "drive_type": np.random.choice(drive_types),
        "body_style": np.random.choice(body_styles),
        "price": round(price, 2),
    })

df = pd.DataFrame(rows)
df.to_csv("sample_vehicle_price.csv", index=False)
print(f"Generated sample_vehicle_price.csv with {len(df)} rows")
print(df.head())
print("\nCorrelation check:")
print(df[["year", "mileage", "horsepower", "price"]].corr()["price"])
