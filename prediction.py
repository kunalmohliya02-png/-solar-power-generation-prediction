"""Prediction workflow: loads the saved Multiple Linear Regression model and the
fitted StandardScaler, and predicts AC power. Nothing is re-fitted here."""
from pathlib import Path
import joblib
import pandas as pd

MODEL_DIR = Path(__file__).parent / "models"
# Exact order used in training (stored inside the scaler as feature_names_in_)
FEATURES = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
POWER_UNIT = "kW"  # AC_POWER unit per the original dataset documentation

# Valid input ranges (physical limits, slightly wider than training data:
# ambient 20.4-39.2 C, module 18.1-66.6 C, irradiation 0-1.22 kW/m2)
LIMITS = {
    "AMBIENT_TEMPERATURE": (-10.0, 60.0),
    "MODULE_TEMPERATURE": (-10.0, 90.0),
    "IRRADIATION": (0.0, 1.5),
}
TRAIN_RANGE = {
    "AMBIENT_TEMPERATURE": (20.4, 39.2),
    "MODULE_TEMPERATURE": (18.1, 66.6),
    "IRRADIATION": (0.0, 1.22),
}

_model = _scaler = None


def load_artifacts():
    """Load model + scaler once and check they match the expected feature order."""
    global _model, _scaler
    if _model is None:
        _model = joblib.load(MODEL_DIR / "solar_model.pkl")
        _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        if list(_scaler.feature_names_in_) != FEATURES:
            raise ValueError(f"Scaler feature order {list(_scaler.feature_names_in_)} != {FEATURES}")
    return _model, _scaler


def validate_inputs(ambient, module, irradiation):
    """Return a list of human-readable error messages (empty list = valid)."""
    errors = []
    for name, label, value in zip(FEATURES, ["Ambient temperature", "Module temperature", "Irradiation"],
                                  [ambient, module, irradiation]):
        if value is None or pd.isna(value):
            errors.append(f"{label} is required.")
            continue
        lo, hi = LIMITS[name]
        if not lo <= value <= hi:
            errors.append(f"{label} must be between {lo} and {hi}.")
    return errors


def predict_power(ambient, module, irradiation):
    """Return dict with raw model output and final (clipped at 0) prediction."""
    errors = validate_inputs(ambient, module, irradiation)
    if errors:
        raise ValueError(" ".join(errors))
    model, scaler = load_artifacts()
    X = pd.DataFrame([[ambient, module, irradiation]], columns=FEATURES)
    raw = float(model.predict(scaler.transform(X))[0])  # transform only, never fit
    return {"raw": raw, "power": max(raw, 0.0)}  # negative power is not physical
