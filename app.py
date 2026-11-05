import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Digital Marketing Conversion Predictor",
    page_icon="📈",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "digital_marketing_conversion_model.pkl"

# ---------------------------------------------------------
# 2. CUSTOM PAGE STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f7f8fc;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(115deg, #172554 0%, #334e8c 55%, #5377b8 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        color: white;
        margin-bottom: 0.35rem;
        font-size: 2.1rem;
    }
    .hero p {
        color: #e5edff;
        margin-bottom: 0;
        font-size: 1rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e7eaf2;
        padding: 14px 16px;
        border-radius: 12px;
    }
    .section-note {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: -0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. LOAD THE TRAINED MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

artifact = load_model_artifact()

st.markdown("""
<div class="hero">
    <h1>📈 Digital Marketing Campaign Conversion Predictor</h1>
    <p>Estimate the likelihood that a customer will convert based on their
    demographics, campaign details, and engagement activity.</p>
</div>
""", unsafe_allow_html=True)

if artifact is None:
    st.error(
        "Model file not found. Place "
        "'digital_marketing_conversion_model.pkl' in the same folder as app.py."
    )
    st.info(
        "Run the final model-saving cell in your Digital Marketing notebook first."
    )
    st.stop()

# The notebook saves a dictionary containing the pipeline and metadata.
# This also supports loading a pipeline directly, if you saved it that way.
if isinstance(artifact, dict) and "model" in artifact:
    model = artifact["model"]
    input_features = artifact.get("input_features", [])
    numeric_features = artifact.get("numeric_features", [])
    categorical_features = artifact.get("categorical_features", [])
    model_name = artifact.get("model_name", "Trained classification model")
else:
    model = artifact
    input_features = list(getattr(model, "feature_names_in_", []))
    numeric_features = []
    categorical_features = []
    model_name = type(model).__name__

# If metadata is unavailable, use the feature names stored by the fitted pipeline.
if not input_features:
    input_features = list(getattr(model, "feature_names_in_", []))

if not input_features:
    st.error(
        "The saved model does not contain its input feature names. "
        "Please save the model using the final cell in the provided notebook."
    )
    st.stop()

# Get exact category options from the fitted OneHotEncoder.
def get_fitted_category_options(fitted_model, cat_columns):
    options = {}
    try:
        preprocessor = fitted_model.named_steps["preprocessor"]
        cat_pipeline = preprocessor.named_transformers_["cat"]
        encoder = cat_pipeline.named_steps["onehot"]
        for column, values in zip(cat_columns, encoder.categories_):
            clean_values = [value for value in values if pd.notna(value)]
            options[column] = [str(value) for value in clean_values]
    except (KeyError, AttributeError, TypeError):
        pass
    return options

category_options = get_fitted_category_options(model, categorical_features)

# Fallback category options for the known project columns if metadata
# cannot be read from the fitted encoder.
fallback_categories = {
    "Gender": ["Female", "Male"],
    "CampaignChannel": ["Email", "Social Media", "Paid Ads", "SEO", "Referral"],
    "CampaignType": ["Awareness", "Consideration", "Conversion", "Retention"]
}

# Defaults are starting values only. Adjust them to realistic values for your dataset.
numeric_defaults = {
    "Age": 30,
    "Income": 60000.0,
    "AdSpend": 2500.0,
    "ClickThroughRate": 0.15,
    "WebsiteVisits": 10,
    "PagesPerVisit": 4.0,
    "TimeOnSite": 8.0,
    "SocialShares": 10,
    "EmailOpens": 5,
    "EmailClicks": 3,
    "PreviousPurchases": 2,
    "LoyaltyPoints": 500
}

integer_features = {
    "Age", "WebsiteVisits", "SocialShares", "EmailOpens",
    "EmailClicks", "PreviousPurchases", "LoyaltyPoints"
}

feature_labels = {
    "Age": "Customer Age",
    "Gender": "Gender",
    "Income": "Annual Income",
    "CampaignChannel": "Campaign Channel",
    "CampaignType": "Campaign Type",
    "AdSpend": "Advertising Spend",
    "ClickThroughRate": "Click-Through Rate",
    "WebsiteVisits": "Website Visits",
    "PagesPerVisit": "Pages per Visit",
    "TimeOnSite": "Time on Site",
    "SocialShares": "Social Shares",
    "EmailOpens": "Email Opens",
    "EmailClicks": "Email Clicks",
    "PreviousPurchases": "Previous Purchases",
    "LoyaltyPoints": "Loyalty Points"
}

# ---------------------------------------------------------
# 4. SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("About this application")
    st.write(
        "This application uses a machine-learning classification model "
        "trained on the Digital Marketing Campaign dataset."
    )
    st.divider()
    st.caption(f"Selected model: {model_name}")
    st.caption("Prediction target: Conversion")
    st.caption("0 = No conversion | 1 = Conversion")
    st.divider()
    st.caption(
        "Enter customer and campaign information, then select "
        "**Predict Conversion**."
    )

# ---------------------------------------------------------
# 5. INPUT FORM
# ---------------------------------------------------------
st.subheader("Enter Customer and Campaign Information")
st.markdown(
    '<p class="section-note">Complete the fields below using the available customer or campaign information.</p>',
    unsafe_allow_html=True
)

user_values = {}

with st.form("conversion_prediction_form"):
    # Demographics
    st.markdown("### 👤 Customer Demographics")
    demo_columns = st.columns(3)

    demo_features = ["Age", "Gender", "Income"]
    for index, feature in enumerate(demo_features):
        if feature not in input_features:
            continue
        with demo_columns[index % 3]:
            if feature in categorical_features:
                choices = category_options.get(
                    feature, fallback_categories.get(feature, ["Unknown"])
                )
                if not choices:
                    choices = fallback_categories.get(feature, ["Unknown"])
                user_values[feature] = st.selectbox(
                    feature_labels.get(feature, feature),
                    choices,
                    key=f"input_{feature}"
                )
            else:
                default = numeric_defaults.get(feature, 0.0)
                if feature in integer_features:
                    user_values[feature] = st.number_input(
                        feature_labels.get(feature, feature),
                        min_value=0,
                        value=int(default),
                        step=1,
                        key=f"input_{feature}"
                    )
                else:
                    user_values[feature] = st.number_input(
                        feature_labels.get(feature, feature),
                        min_value=0.0,
                        value=float(default),
                        step=1000.0 if feature == "Income" else 1.0,
                        key=f"input_{feature}"
                    )

    st.divider()
    st.markdown("### 📣 Campaign Details")
    campaign_features = ["CampaignChannel", "CampaignType", "AdSpend", "ClickThroughRate"]
    campaign_columns = st.columns(2)

    for index, feature in enumerate(campaign_features):
        if feature not in input_features:
            continue
        with campaign_columns[index % 2]:
            if feature in categorical_features:
                choices = category_options.get(
                    feature, fallback_categories.get(feature, ["Unknown"])
                )
                if not choices:
                    choices = fallback_categories.get(feature, ["Unknown"])
                user_values[feature] = st.selectbox(
                    feature_labels.get(feature, feature),
                    choices,
                    key=f"input_{feature}"
                )
            else:
                default = numeric_defaults.get(feature, 0.0)
                user_values[feature] = st.number_input(
                    feature_labels.get(feature, feature),
                    min_value=0.0,
                    max_value=1.0 if feature == "ClickThroughRate" else 10000000.0,
                    value=float(default),
                    step=0.01 if feature == "ClickThroughRate" else 100.0,
                    format="%.3f" if feature == "ClickThroughRate" else "%.2f",
                    key=f"input_{feature}"
                )

    st.divider()
    st.markdown("### 🌐 Customer Engagement and Activity")
    engagement_features = [
        "WebsiteVisits", "PagesPerVisit", "TimeOnSite", "SocialShares",
        "EmailOpens", "EmailClicks", "PreviousPurchases", "LoyaltyPoints"
    ]
    engagement_columns = st.columns(4)

    for index, feature in enumerate(engagement_features):
        if feature not in input_features:
            continue
        with engagement_columns[index % 4]:
            default = numeric_defaults.get(feature, 0)
            if feature in integer_features:
                user_values[feature] = st.number_input(
                    feature_labels.get(feature, feature),
                    min_value=0,
                    value=int(default),
                    step=1,
                    key=f"input_{feature}"
                )
            else:
                user_values[feature] = st.number_input(
                    feature_labels.get(feature, feature),
                    min_value=0.0,
                    value=float(default),
                    step=0.5,
                    key=f"input_{feature}"
                )

    st.divider()
    submitted = st.form_submit_button(
        "🔍 Predict Conversion",
        use_container_width=True,
        type="primary"
    )

# ---------------------------------------------------------
# 6. PREDICTION
# ---------------------------------------------------------
if submitted:
    # Ensure the input columns and their order exactly match model training.
    input_data = pd.DataFrame(
        [{column: user_values[column] for column in input_features}],
        columns=input_features
    )

    # Convert categorical values to the same string representation used by
    # the training data, while preserving numeric columns as numeric.
    for column in categorical_features:
        if column in input_data.columns:
            input_data[column] = input_data[column].astype(str)

    try:
        prediction = int(model.predict(input_data)[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)
            if 1 in classes:
                conversion_probability = float(probabilities[classes.index(1)])
            else:
                conversion_probability = float(probabilities[-1])
        else:
            conversion_probability = None

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        if prediction == 1:
            result_col1.success("### Likely to Convert\nThe model predicts a conversion.")
        else:
            result_col1.warning("### Unlikely to Convert\nThe model predicts no conversion.")

        if conversion_probability is not None:
            result_col2.metric(
                "Estimated Conversion Probability",
                f"{conversion_probability * 100:.1f}%"
            )
            st.progress(max(0.0, min(1.0, conversion_probability)))

            st.caption(
                "This is the model's estimated probability, not a guarantee "
                "that the customer will or will not convert."
            )

        with st.expander("View submitted customer information"):
            st.dataframe(input_data, use_container_width=True)

        # Download the submitted record and prediction.
        result_download = input_data.copy()
        result_download["PredictedConversion"] = prediction
        if conversion_probability is not None:
            result_download["ConversionProbability"] = conversion_probability

        st.download_button(
            "⬇️ Download Prediction",
            data=result_download.to_csv(index=False).encode("utf-8"),
            file_name="conversion_prediction.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error("The prediction could not be generated.")
        st.exception(error)

# ---------------------------------------------------------
# 7. FOOTER
# ---------------------------------------------------------
st.divider()
st.caption(
    "Digital Marketing Campaign Conversion Prediction | "
    "Machine Learning Capstone Project"
)
