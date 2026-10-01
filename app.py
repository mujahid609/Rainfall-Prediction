import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Rainfall Prediction System",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model (1) (1).joblib"
SCALER_PATH = BASE_DIR / "scaler (1) (2).joblib"

WIND_DIRS = ["E","ENE","ESE","N","NE","NNE","NNW","NW","S","SE","SSE","SSW","SW","W","WNW","WSW"]
LOCATIONS = [
    "Adelaide","Albany","Albury","AliceSprings","BadgerysCreek","Ballarat","Bendigo",
    "Brisbane","Cairns","Canberra","Cobar","CoffsHarbour","Dartmoor","Darwin",
    "Dubbo","EastSale","GoldCoast","Hobart","Katherine","Launceston","Melbourne",
    "MelbourneAirport","Mildura","Moree","MountGambier","MountGinini","Newcastle",
    "Nhil","NorahHead","NorfolkIsland","Nuriootpa","PearceRAAF","Penrith","Perth",
    "PerthAirport","Portland","Richmond","Sale","SalmonGums","Sydney","SydneyAirport",
    "Townsville","Tuggeranong","Uluru","WaggaWagga","Walpole","Watsonia","Williamtown","Witchcliffe","Wollongong","Woomera"
]

@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def load_feature_options(scaler):
    """Read categorical options from the fitted scaler schema."""
    expected = list(getattr(scaler, "feature_names_in_", []))
    if not expected:
        raise ValueError("The saved scaler does not contain feature_names_in_.")

    def options_for(prefix, fallback):
        names = [x[len(prefix) + 1:] for x in expected if x.startswith(prefix + "_")]
        return list(dict.fromkeys([fallback[0]] + names))

    return (
        options_for("WindGustDir", WIND_DIRS),
        options_for("WindDir9am", WIND_DIRS),
        options_for("WindDir3pm", WIND_DIRS),
        options_for("Location", LOCATIONS),
    )


def build_features(values, expected_features):
    # Initialize the exact schema expected by the fitted scaler.
    row = {feature: 0.0 for feature in expected_features}

    numeric_values = {
        "MinTemp": values["MinTemp"],
        "MaxTemp": values["MaxTemp"],
        "Rainfall": values["Rainfall"],
        "WindGustSpeed": values["WindGustSpeed"],
        "WindSpeed9am": values["WindSpeed9am"],
        "WindSpeed3pm": values["WindSpeed3pm"],
        "Humidity9am": values["Humidity9am"],
        "Humidity3pm": values["Humidity3pm"],
        "Pressure9am": values["Pressure9am"],
        "Pressure3pm": values["Pressure3pm"],
        "Temp9am": values["Temp9am"],
        "Temp3pm": values["Temp3pm"],
        "TempDiff": values["MaxTemp"] - values["MinTemp"],
        "HumidityDiff": values["Humidity9am"] - values["Humidity3pm"],
    }

    for feature, value in numeric_values.items():
        if feature in row:
            row[feature] = value

    # Reproduce one-hot encoding using the exact trained feature names.
    for col in ["WindGustDir", "WindDir9am", "WindDir3pm", "Location"]:
        selected = values[col]
        prefix = col + "_"
        for feature in expected_features:
            if feature.startswith(prefix):
                row[feature] = int(selected == feature[len(prefix):])

    return pd.DataFrame([row], columns=expected_features)

def main():
    st.markdown(
        """
        <style>
        .main-title {font-size: 2.7rem; font-weight: 800; margin-bottom: 0.2rem;}
        .subtitle {font-size: 1.05rem; color: #64748b; margin-bottom: 1.5rem;}
        .result-card {padding: 1.2rem 1.4rem; border-radius: 16px; border: 1px solid #dbeafe; background: #f8fbff;}
        .metric-label {font-size: .9rem; color: #64748b;}
        .metric-value {font-size: 1.8rem; font-weight: 750;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">🌧️ Rainfall Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Machine-learning powered prediction of whether rainfall is expected tomorrow.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("About the model")
        st.write("This interface uses the trained XGBoost model and scaler from your Rainfall Prediction project.")
        st.divider()
        st.caption("Target: RainTomorrow")
        st.caption("Output: Yes / No")
        st.caption("Model: XGBoost")
        st.caption("Preprocessing: StandardScaler + one-hot encoding")

    try:
        model, scaler = load_artifacts()
        expected_features = list(scaler.feature_names_in_)
        WIND_GUST_DIRS, WIND_9_DIRS, WIND_3_DIRS, LOCATIONS = load_feature_options(scaler)
    except Exception as exc:
        st.error(f"Unable to load the trained model files: {exc}")
        st.stop()

    st.subheader("Enter weather conditions")

    c1, c2, c3 = st.columns(3)
    with c1:
        min_temp = st.number_input("Minimum Temperature (°C)", -10.0, 50.0, 12.0, 0.1)
        max_temp = st.number_input("Maximum Temperature (°C)", -10.0, 55.0, 24.0, 0.1)
        rainfall = st.number_input("Rainfall Today (mm)", 0.0, 500.0, 0.0, 0.1)
        humidity_9 = st.slider("Humidity 9am (%)", 0, 100, 70)
    with c2:
        humidity_3 = st.slider("Humidity 3pm (%)", 0, 100, 50)
        pressure_9 = st.number_input("Pressure 9am (hPa)", 980.0, 1050.0, 1015.0, 0.1)
        pressure_3 = st.number_input("Pressure 3pm (hPa)", 980.0, 1050.0, 1012.0, 0.1)
        wind_gust = st.number_input("Wind Gust Speed (km/h)", 0.0, 200.0, 35.0, 1.0)
    with c3:
        wind_9 = st.number_input("Wind Speed 9am (km/h)", 0.0, 150.0, 15.0, 1.0)
        wind_3 = st.number_input("Wind Speed 3pm (km/h)", 0.0, 150.0, 20.0, 1.0)
        temp_9 = st.number_input("Temperature 9am (°C)", -10.0, 50.0, 18.0, 0.1)
        temp_3 = st.number_input("Temperature 3pm (°C)", -10.0, 55.0, 23.0, 0.1)

    st.subheader("Wind & location")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gust_dir = st.selectbox("Wind Gust Direction", WIND_DIRS)
    with c2:
        dir_9 = st.selectbox("Wind Direction 9am", WIND_DIRS)
    with c3:
        dir_3 = st.selectbox("Wind Direction 3pm", WIND_DIRS)
    with c4:
        location = st.selectbox("Location", LOCATIONS)

    st.divider()

    if st.button("🔮 Predict Rainfall", type="primary", use_container_width=True):
        values = {
            "MinTemp": min_temp, "MaxTemp": max_temp, "Rainfall": rainfall,
            "WindGustSpeed": wind_gust, "WindSpeed9am": wind_9, "WindSpeed3pm": wind_3,
            "Humidity9am": humidity_9, "Humidity3pm": humidity_3,
            "Pressure9am": pressure_9, "Pressure3pm": pressure_3,
            "Temp9am": temp_9, "Temp3pm": temp_3,
            "WindGustDir": gust_dir, "WindDir9am": dir_9, "WindDir3pm": dir_3,
            "Location": location,
        }

        X = build_features(values, expected_features)

        try:
            X_scaled = scaler.transform(X)
            prediction = int(model.predict(X_scaled)[0])
            probability = float(model.predict_proba(X_scaled)[0][1]) if hasattr(model, "predict_proba") else None
        except Exception as exc:
            st.error(
                "Prediction could not be generated. The saved model/scaler feature schema may differ "
                f"from the frontend schema. Technical detail: {exc}"
            )
            st.stop()

        st.subheader("Prediction Result")
        if prediction == 1:
            st.warning("🌧️ Rain is predicted for tomorrow.")
        else:
            st.success("☀️ Rain is not predicted for tomorrow.")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Prediction", "Rain" if prediction else "No Rain")
        with r2:
            st.metric("Rain Probability", f"{probability:.1%}" if probability is not None else "N/A")
        with r3:
            st.metric("Location", location)

        if probability is not None:
            st.progress(probability, text=f"Estimated rain probability: {probability:.1%}")

    with st.expander("Model inputs used"):
        st.write(
            "The interface reproduces the notebook preprocessing: numerical weather variables, "
            "TempDiff, HumidityDiff, and one-hot encoded WindGustDir, WindDir9am, WindDir3pm and Location."
        )

if __name__ == "__main__":
    main()
