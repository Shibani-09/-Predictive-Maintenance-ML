import hashlib
import math
import re

import matplotlib.pyplot as plt
import streamlit as st

from streamlit_app.services.ml_service import (
    get_available_engines,
    load_uploaded_sensor_csv,
    load_test_dataset,
    predict_uploaded_engines,
    predict_engine_rul,
    validate_uploaded_sensor_data,
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


def show_csv_upload_prediction() -> None:
    st.subheader("Upload new engine sensor data")
    st.caption(
        "Upload a headered raw C-MAPSS CSV with engine IDs, cycles, settings, "
        "and all 21 sensor columns. Each engine must include at least five "
        "observations with consecutive latest cycles."
    )
    uploaded_file = st.file_uploader(
        "Choose a raw sensor CSV file",
        type=["csv"],
        key="engine_prediction_csv_upload",
    )

    if uploaded_file is None:
        return

    content = uploaded_file.getvalue()
    upload_signature = hashlib.sha256(content).hexdigest()
    if st.session_state.get("uploaded_csv_signature") != upload_signature:
        st.session_state.pop("uploaded_csv_prediction_results", None)
        st.session_state["uploaded_csv_signature"] = upload_signature

    try:
        uploaded_data = load_uploaded_sensor_csv(uploaded_file.name, content)
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"The uploaded CSV could not be read: {error}")
        return

    st.caption(
        f"Dataset preview · {len(uploaded_data):,} rows · "
        f"{len(uploaded_data.columns):,} columns"
    )
    st.dataframe(
        uploaded_data.head(10),
        use_container_width=True,
        hide_index=True,
    )

    try:
        validated_data = validate_uploaded_sensor_data(uploaded_data)
    except ValueError as error:
        st.error(f"CSV validation failed: {error}")
        return

    st.success(
        f"CSV validated: {validated_data['engine_id'].nunique()} engines "
        f"and {len(validated_data):,} sensor records."
    )

    if st.button(
        "Predict RUL for uploaded engines",
        type="primary",
        key="predict_uploaded_csv",
    ):
        try:
            results = predict_uploaded_engines(validated_data)
            classifications = results["predicted_rul"].apply(
                classify_engine_health
            )
            results["health_status"] = classifications.apply(lambda value: value[0])
            results["recommendation"] = classifications.apply(
                lambda value: value[1]
            )
            st.session_state["uploaded_csv_prediction_results"] = results
        except (OSError, RuntimeError, ValueError) as error:
            st.session_state.pop("uploaded_csv_prediction_results", None)
            st.error(f"Uploaded-data prediction failed: {error}")

    results = st.session_state.get("uploaded_csv_prediction_results")
    if results is not None:
        st.subheader("Uploaded-data RUL predictions")
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download prediction results as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="uploaded_engine_rul_predictions.csv",
            mime="text/csv",
            key=f"download_uploaded_predictions_{upload_signature[:12]}",
        )


def show_engine_prediction() -> None:
    st.title("Engine Prediction")
    st.caption(
        "RUL predictions use the saved Random Forest model and the latest "
        "processed NASA C-MAPSS FD001 record for the selected engine."
    )

    show_csv_upload_prediction()
    st.divider()

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
