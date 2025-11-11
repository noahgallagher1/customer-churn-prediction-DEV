"""
Customer Churn Prediction Dashboard

A streamlined, business-focused dashboard that identifies high-risk customers,
explains churn drivers, and guides retention strategy.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from pathlib import Path
import sys

# Add src to path for config and ui_text import
sys.path.insert(0, str(Path(__file__).parent / 'src'))
import config
import ui_text

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# STYLING
# =============================================================================

st.markdown("""
<style>
    /* Full width layout */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    .main .block-container {
        max-width: 100% !important;
    }

    /* Risk level boxes */
    .risk-high {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }

    .risk-medium {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }

    .risk-low {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }

    /* Section spacing */
    .section-break {
        margin: 3rem 0 2rem 0;
    }

    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts."""
    try:
        model = joblib.load(config.MODEL_FILE)
        preprocessor = joblib.load(config.PREPROCESSOR_FILE)
        feature_names = joblib.load(config.FEATURE_NAMES_FILE)
        metrics = joblib.load(config.METRICS_FILE)

        # Load SHAP objects if available
        try:
            possible_paths = [
                config.MODELS_DIR / 'shap_objects.joblib',
                Path('models/shap_objects.joblib'),
                Path('./models/shap_objects.joblib'),
            ]
            shap_data = None
            for path in possible_paths:
                if path.exists():
                    shap_data = joblib.load(path)
                    break
        except Exception as e:
            print(f"Could not load SHAP data: {e}")
            shap_data = None

        return model, preprocessor, feature_names, metrics, shap_data
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.info("Please run the training pipeline first: python src/model_training.py")
        return None, None, None, None, None


@st.cache_data
def load_test_data():
    """Load test dataset."""
    try:
        test_data = pd.read_csv(config.TEST_DATA_FILE)
        return test_data
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None


def get_feature_display_name(feature_name):
    """Map technical feature names to business-friendly names."""
    mapping = {
        'Contract_Month-to-month': 'Month-to-month Contract',
        'Contract_One year': 'One Year Contract',
        'Contract_Two year': 'Two Year Contract',
        'InternetService_DSL': 'DSL Internet',
        'InternetService_Fiber optic': 'Fiber Optic Internet',
        'InternetService_No': 'No Internet Service',
        'PaymentMethod_Bank transfer (automatic)': 'Bank Transfer (Auto)',
        'PaymentMethod_Credit card (automatic)': 'Credit Card (Auto)',
        'PaymentMethod_Electronic check': 'Electronic Check',
        'PaymentMethod_Mailed check': 'Mailed Check',
        'OnlineSecurity_Yes': 'Has Online Security',
        'OnlineSecurity_No': 'No Online Security',
        'OnlineBackup_Yes': 'Has Online Backup',
        'OnlineBackup_No': 'No Online Backup',
        'DeviceProtection_Yes': 'Has Device Protection',
        'DeviceProtection_No': 'No Device Protection',
        'TechSupport_Yes': 'Has Tech Support',
        'TechSupport_No': 'No Tech Support',
        'StreamingTV_Yes': 'Has Streaming TV',
        'StreamingTV_No': 'No Streaming TV',
        'StreamingMovies_Yes': 'Has Streaming Movies',
        'StreamingMovies_No': 'No Streaming Movies',
        'tenure': 'Tenure (months)',
        'MonthlyCharges': 'Monthly Charges',
        'TotalCharges': 'Total Charges',
        'SeniorCitizen': 'Senior Citizen',
        'Partner_Yes': 'Has Partner',
        'Partner_No': 'No Partner',
        'Dependents_Yes': 'Has Dependents',
        'Dependents_No': 'No Dependents',
        'PaperlessBilling_Yes': 'Paperless Billing',
        'PaperlessBilling_No': 'Paper Billing',
        'charges_per_tenure': 'Avg Monthly Charges',
        'total_services': 'Total Services Subscribed',
    }
    return mapping.get(feature_name, feature_name)


def create_segmentation_chart(test_data):
    """Create churn segmentation visualizations by key attributes."""

    # Prepare data
    X_test = test_data.drop(config.TARGET_COLUMN, axis=1)
    y_test = test_data[config.TARGET_COLUMN]
    df = X_test.copy()
    df['Churn'] = y_test

    # Create segmentation data for visualization
    segments = []

    # Contract Type Segmentation
    if 'Contract_Month-to-month' in df.columns:
        for contract in ['Month-to-month', 'One year', 'Two year']:
            col = f'Contract_{contract}'
            if col in df.columns:
                segment_df = df[df[col] == 1]
                if len(segment_df) > 0:
                    churn_rate = segment_df['Churn'].mean()
                    segments.append({
                        'Segment Type': 'Contract',
                        'Segment': contract,
                        'Churn Rate': churn_rate,
                        'Customer Count': len(segment_df)
                    })

    # Payment Method Segmentation
    payment_methods = ['Electronic check', 'Mailed check',
                      'Bank transfer (automatic)', 'Credit card (automatic)']
    for payment in payment_methods:
        col = f'PaymentMethod_{payment}'
        if col in df.columns:
            segment_df = df[df[col] == 1]
            if len(segment_df) > 0:
                churn_rate = segment_df['Churn'].mean()
                segments.append({
                    'Segment Type': 'Payment Method',
                    'Segment': payment.replace(' (automatic)', ' (auto)'),
                    'Churn Rate': churn_rate,
                    'Customer Count': len(segment_df)
                })

    # Tenure Segmentation
    if 'tenure' in df.columns:
        bins = [0, 12, 24, 48, 100]
        labels = ['0-12 months', '13-24 months', '25-48 months', '49+ months']
        df['Tenure Group'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)

        for group in labels:
            segment_df = df[df['Tenure Group'] == group]
            if len(segment_df) > 0:
                churn_rate = segment_df['Churn'].mean()
                segments.append({
                    'Segment Type': 'Tenure',
                    'Segment': group,
                    'Churn Rate': churn_rate,
                    'Customer Count': len(segment_df)
                })

    return pd.DataFrame(segments)


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

# Load model artifacts
model, preprocessor, feature_names, metrics, shap_data = load_model_artifacts()

if model is None:
    st.error("⚠️ Model artifacts not found. Please run the training pipeline first.")
    st.stop()

# =============================================================================
# PAGE TITLE & OVERVIEW
# =============================================================================

st.title(ui_text.PAGE_TITLE)
st.markdown(ui_text.PAGE_OVERVIEW)
st.markdown("---")

# =============================================================================
# SECTION 1: BUSINESS CONTEXT
# =============================================================================

with st.expander(ui_text.BUSINESS_CONTEXT_TITLE):
    st.write(ui_text.BUSINESS_CONTEXT_CONTENT)

st.markdown("###")

# =============================================================================
# SECTION 2: CHURN DRIVER INSIGHTS (SHAP)
# =============================================================================

st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)
st.subheader(ui_text.CHURN_DRIVERS_TITLE)
st.write(ui_text.CHURN_DRIVERS_INTRO)

# Display SHAP feature importance
try:
    feature_importance_df = pd.read_csv(config.REPORTS_DIR / 'shap_feature_importance.csv')

    # Map to business-friendly names and group by feature category
    feature_importance_df['display_name'] = feature_importance_df['feature'].apply(get_feature_display_name)

    # Group one-hot encoded features
    grouped_importance = feature_importance_df.groupby('display_name').agg({
        'importance': 'sum'
    }).reset_index()

    grouped_importance = grouped_importance.sort_values('importance', ascending=False)

    col1, col2 = st.columns([3, 2])

    with col1:
        # Top features bar chart
        top_n = 10
        top_features = grouped_importance.head(top_n)

        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['display_name'],
            orientation='h',
            marker=dict(
                color=top_features['importance'],
                colorscale='Blues',
                showscale=False
            ),
            text=[f"{val:.3f}" for val in top_features['importance']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=f"Top {top_n} Churn Drivers",
            xaxis_title="Impact on Churn Prediction",
            yaxis_title="",
            height=450,
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(ui_text.CHURN_DRIVERS_INSIGHTS)

except Exception as e:
    st.warning("⚠️ SHAP feature importance visualizations not available.")
    st.info(f"Error: {e}")

st.markdown("###")

# =============================================================================
# SECTION 3: CHURN SEGMENTATION
# =============================================================================

st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)
st.subheader(ui_text.CHURN_SEGMENTATION_TITLE)
st.write(ui_text.CHURN_SEGMENTATION_INTRO)

# Load test data and create segmentation chart
test_data = load_test_data()

if test_data is not None:
    segments_df = create_segmentation_chart(test_data)

    if not segments_df.empty:
        # Create faceted bar charts by segment type
        segment_types = segments_df['Segment Type'].unique()

        cols = st.columns(len(segment_types))

        for idx, segment_type in enumerate(segment_types):
            with cols[idx]:
                seg_data = segments_df[segments_df['Segment Type'] == segment_type]
                seg_data = seg_data.sort_values('Churn Rate', ascending=False)

                fig = go.Figure(go.Bar(
                    x=seg_data['Segment'],
                    y=seg_data['Churn Rate'] * 100,
                    marker=dict(
                        color=seg_data['Churn Rate'] * 100,
                        colorscale='Reds',
                        showscale=False
                    ),
                    text=[f"{val:.1f}%" for val in seg_data['Churn Rate'] * 100],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1f}%<br>Customers: %{customdata}<extra></extra>',
                    customdata=seg_data['Customer Count']
                ))

                fig.update_layout(
                    title=f"{segment_type}",
                    xaxis_title="",
                    yaxis_title="Churn Rate (%)",
                    height=350,
                    template='plotly_white',
                    showlegend=False
                )

                fig.update_xaxes(tickangle=-45)

                st.plotly_chart(fig, use_container_width=True)

        st.markdown("###")

st.markdown("###")

# =============================================================================
# SECTION 4: CUSTOMER RISK LOOKUP
# =============================================================================

st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)
st.subheader(ui_text.CUSTOMER_RISK_TITLE)
st.write(ui_text.CUSTOMER_RISK_INTRO)

if test_data is not None:
    X_test = test_data.drop(config.TARGET_COLUMN, axis=1)
    y_test = test_data[config.TARGET_COLUMN]

    # Customer selection
    col_select1, col_select2 = st.columns([1, 3])

    with col_select1:
        customer_idx = st.number_input(
            "Customer ID (Index)",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0,
            help="Enter a customer index to analyze their churn risk"
        )

    with col_select2:
        st.info(f"Analyzing customer at index {customer_idx} from the test dataset")

    # Get customer data
    customer_data = X_test.iloc[customer_idx:customer_idx+1]
    actual_churn = y_test.iloc[customer_idx]

    # Make prediction
    prediction_proba = model.predict_proba(customer_data)[0]
    churn_probability = prediction_proba[1]
    prediction = "CHURN" if churn_probability >= 0.5 else "NO CHURN"

    # Determine risk level
    if churn_probability >= 0.7:
        risk_level = ui_text.RISK_HIGH
        risk_desc = ui_text.RISK_HIGH_DESC
        risk_class = "risk-high"
        risk_emoji = "🔴"
    elif churn_probability >= 0.4:
        risk_level = ui_text.RISK_MEDIUM
        risk_desc = ui_text.RISK_MEDIUM_DESC
        risk_class = "risk-medium"
        risk_emoji = "🟡"
    else:
        risk_level = ui_text.RISK_LOW
        risk_desc = ui_text.RISK_LOW_DESC
        risk_class = "risk-low"
        risk_emoji = "🟢"

    st.markdown("###")

    # Display results
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f'<div class="{risk_class}"><h3>{risk_emoji} {risk_level}</h3><p>{risk_desc}</p></div>',
                   unsafe_allow_html=True)

    with col2:
        st.metric("Churn Probability", f"{churn_probability:.1%}")

    with col3:
        st.metric("Prediction", prediction)

    with col4:
        actual_label = "CHURN" if actual_churn == 1 else "NO CHURN"
        st.metric("Actual Status", actual_label)
        correct = "✓ Correct" if (prediction == actual_label) else "✗ Incorrect"
        st.caption(correct)

    st.markdown("###")

    # SHAP Explanation for this customer
    if shap_data is not None and shap_data.get('explainer') is not None:
        st.markdown(f"**{ui_text.SHAP_EXPLANATION_HEADER}**")

        explainer = shap_data['explainer']

        try:
            # Calculate SHAP values
            customer_shap = explainer.shap_values(customer_data)
            if isinstance(customer_shap, list):
                customer_shap = customer_shap[1]

            # Create SHAP waterfall plot
            fig, ax = plt.subplots(figsize=(12, 8))

            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1]

            shap_exp = shap.Explanation(
                values=customer_shap[0],
                base_values=expected_value,
                data=customer_data.iloc[0].values,
                feature_names=customer_data.columns.tolist()
            )

            shap.plots.waterfall(shap_exp, show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Top drivers text summary
            st.markdown(f"**{ui_text.TOP_DRIVERS_HEADER}**")

            # Get top features
            feature_impacts = list(zip(customer_data.columns, customer_shap[0], customer_data.iloc[0].values))
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

            top_drivers = feature_impacts[:5]

            for feature, impact, value in top_drivers:
                display_name = get_feature_display_name(feature)
                direction = "increases" if impact > 0 else "decreases"
                st.markdown(f"- **{display_name}** (value: {value:.2f}) {direction} churn risk by {abs(impact):.3f}")

        except Exception as e:
            st.info(f"Could not generate SHAP explanation: {e}")

    st.markdown("###")

    # Recommended Actions
    st.markdown(f"**{ui_text.RECOMMENDED_ACTIONS_HEADER}**")

    if churn_probability >= 0.7:
        actions = ui_text.HIGH_RISK_ACTIONS
        st.markdown(f"""
        <div class="{risk_class}">
        <h4>{actions['title']}</h4>
        <ul>
        """ + "".join([f"<li>{action}</li>" for action in actions['actions']]) + f"""
        </ul>
        <p><b>Estimated Cost:</b> {actions['estimated_cost']} | <b>Expected ROI:</b> {actions['expected_roi']}</p>
        </div>
        """, unsafe_allow_html=True)

    elif churn_probability >= 0.4:
        actions = ui_text.MEDIUM_RISK_ACTIONS
        st.markdown(f"""
        <div class="{risk_class}">
        <h4>{actions['title']}</h4>
        <ul>
        """ + "".join([f"<li>{action}</li>" for action in actions['actions']]) + f"""
        </ul>
        <p><b>Estimated Cost:</b> {actions['estimated_cost']} | <b>Success Rate:</b> {actions['success_rate']}</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        actions = ui_text.LOW_RISK_ACTIONS
        st.markdown(f"""
        <div class="{risk_class}">
        <h4>{actions['title']}</h4>
        <ul>
        """ + "".join([f"<li>{action}</li>" for action in actions['actions']]) + f"""
        </ul>
        <p><b>Estimated Cost:</b> {actions['estimated_cost']} | <b>Success Rate:</b> {actions['success_rate']}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("###")

# =============================================================================
# SECTION 5: RETENTION STRATEGY RECOMMENDATIONS
# =============================================================================

st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)
st.subheader(ui_text.RETENTION_STRATEGY_TITLE)
st.write(ui_text.RETENTION_STRATEGY_INTRO)

st.markdown(ui_text.RETENTION_STRATEGY_TABLE)

st.markdown("###")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption(ui_text.MODEL_DISCLAIMER)
