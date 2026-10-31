from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
FEATURE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


@st.cache_resource
def load_model_artifacts() -> tuple[Any, Any, list[str]]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict):
        raise TypeError("The model artifact must be a dictionary.")

    required_keys = {"preprocessor", "model", "features"}
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        raise ValueError(
            "The model artifact is missing required entries: "
            + ", ".join(sorted(missing_keys))
        )

    features = artifact["features"]
    if not isinstance(features, list) or features != FEATURE_COLUMNS:
        raise ValueError(
            "The model's feature order does not match the inputs supported by this app."
        )

    model = artifact["model"]
    if not callable(getattr(model, "predict", None)) or not callable(
        getattr(model, "predict_proba", None)
    ):
        raise TypeError("The saved model must support predict and predict_proba.")

    return artifact["preprocessor"], model, features


def main() -> None:
    st.set_page_config(page_title="Customer Churn Predictor")
    st.title("Customer Churn Predictor")
    st.write(
        "Enter customer details to estimate the likelihood that the customer "
        "will leave the bank."
    )

    try:
        preprocessor, model, features = load_model_artifacts()
    except (FileNotFoundError, TypeError, ValueError) as error:
        st.error(str(error))
        st.stop()

    with st.form("customer_details"):
        left, right = st.columns(2)
        with left:
            credit_score = st.number_input(
                "Credit score", min_value=300, max_value=900, value=650
            )
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", min_value=18, max_value=120, value=40)
            tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
        with right:
            balance = st.number_input(
                "Account balance", min_value=0.0, value=75_000.0, step=1_000.0
            )
            products = st.number_input(
                "Number of products", min_value=1, max_value=4, value=2
            )
            has_card = st.checkbox("Has a credit card", value=True)
            is_active = st.checkbox("Is an active member", value=True)
            salary = st.number_input(
                "Estimated salary", min_value=0.0, value=100_000.0, step=1_000.0
            )

        submitted = st.form_submit_button("Estimate churn risk")

    if submitted:
        customer = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": int(has_card),
            "IsActiveMember": int(is_active),
            "EstimatedSalary": salary,
        }
        customer_frame = pd.DataFrame([customer], columns=features)
        transformed = preprocessor.transform(customer_frame)
        class_labels = list(model.classes_)
        if 1 not in class_labels:
            raise ValueError("The saved model does not provide the churn class (1).")

        churn_probability = float(
            model.predict_proba(transformed)[0][class_labels.index(1)]
        )
        prediction = int(model.predict(transformed)[0])

        if prediction == 1:
            st.warning(f"Estimated churn risk: {churn_probability:.1%}")
        else:
            st.success(f"Estimated churn risk: {churn_probability:.1%}")
        st.caption(
            "This estimate is for demonstration and should not be used as the "
            "sole basis for customer or financial decisions."
        )


if __name__ == "__main__":
    main()
