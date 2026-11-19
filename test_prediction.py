"""Checks the pipeline against the real dataset and the metrics in the report."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from prediction import FEATURES, predict_power, validate_inputs

df = pd.read_csv("Results/solar_power_cleaned.csv")
_, Xte, _, yte = train_test_split(df[FEATURES], df["AC_POWER"], test_size=0.2, random_state=42)
pred = [predict_power(*r)["raw"] for r in Xte.itertuples(index=False)]
r2, mae = r2_score(yte, pred), mean_absolute_error(yte, pred)
rmse = mean_squared_error(yte, pred) ** 0.5
print(f"R2={r2:.5f} MAE={mae:.2f} RMSE={rmse:.2f}")
assert abs(r2 - 0.903094) < 1e-5 and abs(mae - 1438.585) < 0.01 and abs(rmse - 2403.003) < 0.01
assert predict_power(25, 22, 0.0)["power"] >= 0          # night: clipped, never negative
assert predict_power(30, 50, 0.8)["power"] > predict_power(30, 40, 0.4)["power"]
assert validate_inputs(30, 40, 2.0) and validate_inputs(None, 40, 0.5) and not validate_inputs(30, 40, 0.5)
print("ALL TESTS PASSED")
