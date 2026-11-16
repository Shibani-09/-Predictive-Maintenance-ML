
import sys
from pathlib import Path

# --------------------------------------------------
# PROJECT ROOT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if not sys.path or Path(sys.path[0] or ".").resolve() != PROJECT_ROOT:
    if str(PROJECT_ROOT) in sys.path:
        sys.path.remove(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib as mpl

from streamlit_app.services.database import get_database
from streamlit_app.services.authentication import (
    authenticate_user,
    register_user,
)
from streamlit_app.services.ml_service import (
    load_feature_columns,
    load_model,
    load_test_dataset,
)
from streamlit_app.pages.engine_prediction import show_engine_prediction
from streamlit_app.pages.model_performance import show_model_performance
from streamlit_app.pages.realtime_monitoring import show_realtime_monitoring


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

mpl.rcParams.update(
    {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#DFE7F2",
        "axes.labelcolor": "#718096",
        "axes.titlecolor": "#1D2F4F",
        "xtick.color": "#718096",
        "ytick.color": "#718096",
        "text.color": "#1D2F4F",
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Arial"],
    }
)

st.markdown("""
<style>
    :root {
        color-scheme: light;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: #F4F7FC;
        color: #1D2F4F;
    }

    [data-testid="stHeader"] {
        background: rgba(244, 247, 252, 0.92);
        border-bottom: 1px solid rgba(223, 231, 242, 0.75);
    }

    .main .block-container {
        max-width: 1440px;
        padding: 2.4rem 3rem 3.5rem;
    }

    h1, h2, h3, h4 {
        color: #1D2F4F;
        letter-spacing: -0.025em;
    }

    h1 {
        font-weight: 750;
        line-height: 1.2;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #718096;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #092653 0%, #0B3268 100%);
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #F7FAFF;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #B5C9E8;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        min-height: 2.8rem;
        margin: 0.2rem 0;
        padding: 0.55rem 0.8rem;
        border-radius: 0.7rem;
        transition: background-color 160ms ease, transform 160ms ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.09);
        transform: translateX(2px);
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(110deg, #1769E8 0%, #3486F2 100%);
        box-shadow: 0 7px 18px rgba(2, 20, 52, 0.24);
    }

    [data-testid="stSidebar"] [data-testid="stButton"] button {
        color: #FFFFFF;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.22);
    }

    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background: rgba(255, 255, 255, 0.18);
        border-color: rgba(255, 255, 255, 0.42);
    }

    div[data-testid="stMetric"] {
        min-height: 112px;
        padding: 1.15rem 1.3rem;
        background: #FFFFFF;
        border: 1px solid #DFE7F2;
        border-left: 4px solid #1769E8;
        border-radius: 1rem;
        box-shadow: 0 8px 24px rgba(20, 47, 86, 0.07);
        transition: transform 180ms ease, box-shadow 180ms ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 30px rgba(20, 47, 86, 0.12);
    }

    [data-testid="stMetricLabel"] {
        color: #718096;
        font-size: 0.88rem;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #1D2F4F;
        font-weight: 750;
    }

    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] button {
        min-height: 2.7rem;
        border-radius: 0.72rem;
        font-weight: 650;
        transition: transform 160ms ease, box-shadow 160ms ease,
            background-color 160ms ease, border-color 160ms ease;
    }

    [data-testid="stButton"] button[kind="primary"],
    [data-testid="stFormSubmitButton"] button[kind="primary"] {
        color: #FFFFFF;
        background: linear-gradient(115deg, #1769E8 0%, #244A91 100%);
        border: 1px solid #1769E8;
        box-shadow: 0 7px 16px rgba(23, 105, 232, 0.23);
    }

    [data-testid="stButton"] button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
        background: linear-gradient(115deg, #3486F2 0%, #1769E8 100%);
        border-color: #1769E8;
        box-shadow: 0 10px 22px rgba(23, 105, 232, 0.3);
        transform: translateY(-1px);
    }

    [data-testid="stButton"] button:not([kind="primary"]),
    [data-testid="stFormSubmitButton"] button:not([kind="primary"]) {
        color: #244A91;
        background: #FFFFFF;
        border: 1px solid #DFE7F2;
    }

    [data-testid="stButton"] button:not([kind="primary"]):hover,
    [data-testid="stFormSubmitButton"] button:not([kind="primary"]):hover {
        color: #1769E8;
        border-color: #1769E8;
        box-shadow: 0 6px 16px rgba(23, 105, 232, 0.12);
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        color: #1D2F4F;
        background: #FFFFFF;
        border-color: #DFE7F2;
        border-radius: 0.7rem;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: #1769E8;
        box-shadow: 0 0 0 3px rgba(23, 105, 232, 0.12);
    }

    [data-testid="stForm"],
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #DFE7F2;
        border-radius: 1rem;
        box-shadow: 0 8px 24px rgba(20, 47, 86, 0.06);
    }

    [data-testid="stForm"] {
        padding: 1.25rem 1.4rem;
    }

    [data-testid="stExpander"] summary {
        color: #1D2F4F;
        font-weight: 650;
    }

    [data-baseweb="tab-list"] {
        gap: 0.45rem;
        border-bottom: 1px solid #DFE7F2;
    }

    [data-testid="stTab"] {
        color: #718096;
        background: transparent;
        border-radius: 0.65rem 0.65rem 0 0;
        font-weight: 650;
        transition: color 160ms ease, background-color 160ms ease;
    }

    [data-testid="stTab"] p {
        color: inherit;
    }

    [data-testid="stTab"][aria-selected="true"] {
        color: #1769E8;
        background: #E5EEFF;
        border-color: #1769E8;
    }

    [data-testid="stTab"][aria-selected="true"] .react-aria-SelectionIndicator {
        background: #1769E8;
    }

    [data-testid="stAlert"] {
        border-radius: 0.85rem;
        border: 1px solid #DFE7F2;
        box-shadow: 0 5px 16px rgba(20, 47, 86, 0.04);
    }

    [data-testid="stPyplot"],
    [data-testid="stPlotlyChart"],
    [data-testid="stImage"] {
        overflow: hidden;
        padding: 0.5rem;
        background: #FFFFFF;
        border: 1px solid #DFE7F2;
        border-radius: 1rem;
        box-shadow: 0 8px 24px rgba(20, 47, 86, 0.07);
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        overflow: hidden;
        border: 1px solid #DFE7F2;
        border-radius: 0.85rem;
        box-shadow: 0 6px 18px rgba(20, 47, 86, 0.05);
    }

    hr {
        border-color: #DFE7F2;
    }

    @media (max-width: 900px) {
        .main .block-container {
            padding: 1.8rem 1.4rem 2.5rem;
        }
    }

    @media (max-width: 600px) {
        .main .block-container {
            padding: 1.35rem 0.9rem 2rem;
        }

        div[data-testid="stMetric"] {
            min-height: auto;
            padding: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user" not in st.session_state:
    st.session_state.user = None


# --------------------------------------------------
# AUTHENTICATION PAGE
# --------------------------------------------------

def show_authentication():

    st.title("⚙️ Predictive Maintenance")
    st.caption("Industrial Equipment Health Monitoring System")

    st.divider()

    login_tab, register_tab = st.tabs(
        ["Login", "Create Account"]
    )

    # LOGIN
    with login_tab:

        st.subheader("Welcome Back")
        st.write("Login to access your dashboard.")

        with st.form("login_form"):

            email = st.text_input(
                "Email Address",
                placeholder="Enter your email"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )

            login_button = st.form_submit_button(
                "Login",
                use_container_width=True
            )

            if login_button:

                if not email.strip() or not password:
                    st.error("Please enter email and password.")

                else:
                    try:
                        success, message, user = authenticate_user(
                            email,
                            password
                        )

                        if success:
                            st.session_state.authenticated = True
                            st.session_state.user = user

                            st.success(message)
                            st.rerun()

                        else:
                            st.error(message)

                    except Exception:
                        st.error(
                            "Login failed due to a system error. "
                            "Check the MongoDB configuration and connection."
                        )

    # REGISTRATION
    with register_tab:

        st.subheader("Create New Account")
        st.write("Register to use the predictive maintenance system.")

        with st.form("registration_form"):

            full_name = st.text_input(
                "Full Name",
                placeholder="Enter your full name"
            )

            email = st.text_input(
                "Email Address",
                placeholder="Enter your email"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Minimum 8 characters"
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password"
            )

            register_button = st.form_submit_button(
                "Create Account",
                use_container_width=True
            )

            if register_button:

                if not full_name.strip() or not email.strip():
                    st.error("Please complete all fields.")

                elif not password or not confirm_password:
                    st.error("Please enter and confirm your password.")

                elif password != confirm_password:
                    st.error("Passwords do not match.")

                else:
                    try:
                        success, message = register_user(
                            full_name,
                            email,
                            password
                        )

                        if success:
                            st.success(message)
                            st.info(
                                "Your account is ready. "
                                "Please login using your credentials."
                            )

                        else:
                            st.error(message)

                    except Exception:
                        st.error(
                            "Registration failed due to a system error. "
                            "Check the MongoDB configuration and connection."
                        )

    st.divider()

    st.caption(
        "Predictive Maintenance | Python • Streamlit • MongoDB"
    )


# --------------------------------------------------
# AUTHENTICATED PAGES
# --------------------------------------------------

def show_authenticated_sidebar():
    st.sidebar.title("⚙️ Predictive Maintenance")
    st.sidebar.caption("Industrial Equipment Monitoring")
    st.sidebar.divider()

    user = st.session_state.user
    st.sidebar.write(
        f"Welcome, {user['full_name']}"
    )

    st.sidebar.write(
        user["email"]
    )

    if st.sidebar.button(
        "Logout",
        use_container_width=True
    ):
        st.session_state.authenticated = False
        st.session_state.user = None
        st.rerun()


def show_dashboard():
    show_authenticated_sidebar()

    selected_page = st.sidebar.radio(
        "Navigation",
        [
            "Home Dashboard",
            "Engine Prediction",
            "Model Performance",
            "Realtime Monitoring",
        ],
    )

    if selected_page == "Engine Prediction":
        show_engine_prediction()
        return

    if selected_page == "Model Performance":
        show_model_performance()
        return

    if selected_page == "Realtime Monitoring":
        show_realtime_monitoring()
        return

    # Main heading
    st.title("Predictive Maintenance Dashboard")

    st.caption(
        "Machine Learning Based Industrial Equipment "
        "Health Monitoring and Remaining Useful Life Prediction"
    )

    st.divider()

    # Database connection
    db = None
    user_count = 0
    database_status = "Disconnected"

    try:
        db = get_database()
        user_count = db["users"].count_documents({})
        database_status = "Connected"

    except Exception:
        st.error(
            "MongoDB connection failed. Check the configured secrets "
            "and database availability."
        )

    # Project resources
    model_path = PROJECT_ROOT / "models" / "random_forest_rul_model.pkl"
    feature_path = PROJECT_ROOT / "models" / "feature_columns.pkl"
    train_path = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test_features.csv"

    model_status = "Available" if model_path.exists() else "Missing"
    feature_count = 0
    dataset_rows = 0
    engine_count = 0
    model_error = None
    dataset_error = None

    try:
        load_model()
        feature_count = len(load_feature_columns())
    except Exception as error:
        model_status = "Unavailable"
        model_error = str(error)

    try:
        test_dataset = load_test_dataset()
        dataset_rows = len(test_dataset)
        engine_count = test_dataset["engine_id"].nunique()
    except Exception as error:
        dataset_error = str(error)

    evaluation_metrics = {}
    metrics_path = PROJECT_ROOT / "outputs" / "final_model_metrics.csv"

    if metrics_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_path)

            if {"Metric", "Value"}.issubset(metrics_df.columns):
                evaluation_metrics = {
                    str(row["Metric"]): float(row["Value"])
                    for _, row in metrics_df.iterrows()
                }

        except (OSError, ValueError, TypeError) as error:
            st.warning(f"Saved model metrics could not be read: {error}")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Registered Users", user_count)

    with col2:
        st.metric("MongoDB Status", database_status)

    with col3:
        st.metric("ML Model", model_status)

    with col4:
        st.metric(
            "Test Engines",
            engine_count if not dataset_error else "Unavailable"
        )

    st.divider()

    if model_error:
        st.error(f"Random Forest model could not be loaded: {model_error}")

    if dataset_error:
        st.error(f"Processed test dataset could not be loaded: {dataset_error}")

    # Project overview
    st.subheader("System Overview")

    left, right = st.columns(2)

    with left:

        st.markdown("### 🤖 Machine Learning")

        st.write(
            "The system uses a trained Random Forest model "
            "to predict the Remaining Useful Life (RUL) "
            "of industrial equipment."
        )

        st.write("**Model:** Existing Random Forest Regressor")
        st.write("**Model features:**", feature_count or "Unavailable")
        st.write("**Dataset:** NASA C-MAPSS FD001")
        st.write("**Processed test records:**", dataset_rows or "Unavailable")

    with right:

        st.markdown("### 🗄️ Database")

        st.write(
            "MongoDB Atlas stores registered user accounts "
            "and authentication information."
        )

        st.write(
            f"**Database:** "
            f"{db.name if db is not None else 'Unavailable'}"
        )

        st.write(
            f"**User Collection:** {user_count} documents"
        )

    st.divider()

    if evaluation_metrics:
        st.subheader("Saved Model Performance")

        metric_columns = st.columns(len(evaluation_metrics))

        for column, (metric, value) in zip(
            metric_columns, evaluation_metrics.items()
        ):
            column.metric(metric, f"{value:.3f}")

        st.caption(
            "Metrics are the saved evaluation results for the existing model; "
            "the model was not retrained."
        )

    st.divider()

    # Project resources
    st.subheader("Project Resources")

    resources = {
        "Random Forest Model": model_path,
        "Feature Columns": feature_path,
        "Training Dataset": train_path,
        "Testing Dataset": test_path
    }

    for resource, path in resources.items():

        if path.exists():
            st.success(f"{resource}: Available")

        else:
            st.warning(f"{resource}: Not found")

    st.divider()

    st.caption(
        "Predictive Maintenance System | "
        "Python • Streamlit • MongoDB • Machine Learning"
    )


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if st.session_state.authenticated:
    page = st.navigation(
        [
            st.Page(
                show_dashboard,
                title="Predictive Maintenance",
                default=True
            )
        ],
        position="hidden",
    )

else:
    page = st.navigation(
        [
            st.Page(
                show_authentication,
                title="Login",
                default=True
            )
        ],
        position="hidden",
    )

page.run()
