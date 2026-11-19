import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from prediction import LIMITS, TRAIN_RANGE, POWER_UNIT, predict_power

st.set_page_config(page_title="SolarPower Predictor", page_icon="☀️", layout="centered")

DEFAULTS = {"amb": 30.0, "mod": 40.0, "irr": 0.50}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("result", None)


def reset():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["result"] = None


st.markdown("""<style>
.result-box{background:linear-gradient(135deg,#fff7e0,#ffe2a8);border-radius:14px;padding:1.5rem;
text-align:center;border:1px solid #f5c35a;color:#3a2a00}
.result-box .big{font-size:3rem;font-weight:700;line-height:1.1}
.result-box .lbl{font-size:.9rem;letter-spacing:.08em;text-transform:uppercase;opacity:.75}
</style>""", unsafe_allow_html=True)

st.title("☀️ SolarPower Predictor")
st.caption("Solar Power Generation Prediction · Multiple Linear Regression (R² = 0.903)")

with st.container(border=True):
    st.subheader("Environmental parameters")
    c1, c2, c3 = st.columns(3)
    amb = c1.number_input("Ambient Temperature (°C)", step=0.5, key="amb",
                          help=f"Typical range in training data: {TRAIN_RANGE['AMBIENT_TEMPERATURE'][0]}–{TRAIN_RANGE['AMBIENT_TEMPERATURE'][1]} °C")
    mod = c2.number_input("Module Temperature (°C)", step=0.5, key="mod",
                          help=f"Typical range in training data: {TRAIN_RANGE['MODULE_TEMPERATURE'][0]}–{TRAIN_RANGE['MODULE_TEMPERATURE'][1]} °C")
    irr = c3.number_input("Solar Irradiation (kW/m²)", step=0.05, format="%.3f", key="irr",
                          help="Dataset unit is kW/m² (1.0 = 1000 W/m²). Training data range: 0–1.22")
    b1, b2 = st.columns(2)
    go = b1.button("Predict", type="primary", use_container_width=True)
    b2.button("Clear / Reset", on_click=reset, use_container_width=True)

if go:
    try:
        st.session_state["result"] = (predict_power(amb, mod, irr), (amb, mod, irr))
    except ValueError as e:
        st.session_state["result"] = None
        st.error(f"Invalid input: {e}")
    except FileNotFoundError:
        st.session_state["result"] = None
        st.error("Model files not found. Make sure models/solar_model.pkl and models/scaler.pkl exist.")

res = st.session_state["result"]
if res:
    out, (a, m, i) = res
    st.markdown(f'<div class="result-box"><div class="lbl">Predicted AC Power</div>'
                f'<div class="big">{out["power"]:,.1f} {POWER_UNIT}</div></div>', unsafe_allow_html=True)
    if out["raw"] < 0:
        st.info(f"Model output was {out['raw']:,.1f}; shown as 0 because power cannot be negative.")
    outside = [n for n, v in zip(["Ambient", "Module", "Irradiation"], (a, m, i))
               if not TRAIN_RANGE[{"Ambient": "AMBIENT_TEMPERATURE", "Module": "MODULE_TEMPERATURE",
                                   "Irradiation": "IRRADIATION"}[n]][0] <= v <=
                   TRAIN_RANGE[{"Ambient": "AMBIENT_TEMPERATURE", "Module": "MODULE_TEMPERATURE",
                                "Irradiation": "IRRADIATION"}[n]][1]]
    if outside:
        st.warning(f"{', '.join(outside)} outside the range seen in training data (May–June 2020); "
                   "prediction may be less reliable.")
    with st.expander("Entered parameters"):
        st.table({"Parameter": ["Ambient Temperature", "Module Temperature", "Irradiation"],
                  "Value": [f"{a} °C", f"{m} °C", f"{i} kW/m²"]})
    st.caption("This is a real prediction from the saved trained model (not an example value). "
               "It estimates total plant AC power, as in the training data.")
