import math
import re

import matplotlib.pyplot as plt
import streamlit as st

from services.ml_service import (
    get_available_engines,
    load_test_dataset,
    predict_engine_rul,
)


def classify_engine_health(predicted_rul: float) -> tuple[str, str]:
    """Classify predicted RUL using the documented application thresholds."""
    if predicted_rul <= 30:
        return "Critical", "Immediate maintenance required"
    if predicted_rul <= 60:
        return "Warning", "Schedule maintenance soon"
    if predicted_rul <= 120:
        return "Monitor", "Continue monitoring the engine"
    return "Healthy", "No immediate action indicated"


def show_engine_prediction() -> None:
    st.title("Engine Prediction")
    st.caption(
        "RUL predictions use the saved Random Forest model and the latest "
        "processed NASA C-MAPSS FD001 record for the selected engine."
    )

    try:
        engine_ids = get_available_engines()
        dataset = load_test_dataset()
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Engine data could not be loaded: {error}")
        return

    if not engine_ids:
        st.error("No engine records are available in the processed test data.")
        return

    selected_engine = st.selectbox(
        "Select engine",
        engine_ids,
        format_func=lambda engine_id: f"Engine {engine_id}",
    )

    selected_rows = dataset[dataset["engine_id"] == selected_engine].sort_values(
        "cycle"
    )
    latest_cycle = int(selected_rows["cycle"].iloc[-1])

    if st.button("Predict remaining useful life", type="primary"):
        try:
            result = predict_engine_rul(int(selected_engine))
            if not math.isfinite(result["predicted_rul"]):
                raise ValueError("The model returned a non-finite RUL value.")
            st.session_state["engine_prediction_result"] = result
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Prediction could not be completed: {error}")

    result = st.session_state.get("engine_prediction_result")
    if result and result["engine_id"] == selected_engine:
        health, recommendation = classify_engine_health(
            result["predicted_rul"]
        )
        metric_column, health_column = st.columns(2)
        metric_column.metric(
            "Predicted remaining useful life",
            f"{result['predicted_rul']:.2f} cycles",
        )
        health_column.metric("Engine health", health)
        st.info(recommendation)
        st.caption(
            f"Engine {result['engine_id']} · latest observed cycle "
            f"{result['cycle']}"
        )

    st.subheader("Sensor trend")
    st.caption(
        "Historical readings from the selected engine in the processed "
        "NASA dataset; these are not live hardware measurements."
    )

    sensor_columns = [
        column
        for column in selected_rows.columns
        if re.fullmatch(r"sensor_\d+", column)
    ]
    if not sensor_columns:
        st.warning("No raw sensor columns were found for this engine.")
    else:
        selected_sensor = st.selectbox("Sensor", sensor_columns)
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(
            selected_rows["cycle"],
            selected_rows[selected_sensor],
            color="#1769E8",
        )
        axis.set_xlabel("Cycle")
        axis.set_ylabel(selected_sensor)
        axis.set_title(f"{selected_sensor} readings · Engine {selected_engine}")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    with st.expander("Health classification thresholds"):
        st.markdown(
            "- **Critical:** predicted RUL ≤ 30 cycles\n"
            "- **Warning:** 30 < predicted RUL ≤ 60 cycles\n"
            "- **Monitor:** 60 < predicted RUL ≤ 120 cycles\n"
            "- **Healthy:** predicted RUL > 120 cycles"
        )
        st.caption(
            "These app thresholds follow the existing final-engine/dashboard "
            "classification. The historical simulation script uses a different "
            "monitor threshold; the monitoring page will align to these values."
        )
