"""
UI Text Content for Customer Churn Prediction Dashboard

This module contains all narrative text, section headers, and business-focused
messaging used throughout the dashboard. Centralizing this content makes it
easier to maintain consistent tone and messaging.
"""

# =============================================================================
# PAGE TITLE & OVERVIEW
# =============================================================================

PAGE_TITLE = "📉 Customer Churn Prediction Dashboard"

PAGE_OVERVIEW = """
This dashboard combines machine learning, explainability methods, and business reasoning to
identify customers at risk of churn **and understand why**.

The goal is not only to predict churn — but to support **clear, targeted retention strategy**.

**Use this interface to:**
- Explore patterns behind churn behavior
- Understand which customer attributes drive churn risk
- Evaluate individual churn scores with personalized explanations
- Design actionable retention programs grounded in data insight
"""

# =============================================================================
# SECTION 1: BUSINESS CONTEXT
# =============================================================================

BUSINESS_CONTEXT_TITLE = "💡 Why This Matters"

BUSINESS_CONTEXT_CONTENT = """
Customers who churn represent **lost recurring revenue**, and acquiring new customers is far more
expensive than retaining existing ones. This model helps identify *which customers are at risk* so
retention teams can act **before** churn happens.

The dashboard also highlights **why** churn occurs — enabling strategy, not just prediction.
"""

# =============================================================================
# SECTION 2: CHURN DRIVER INSIGHTS (SHAP)
# =============================================================================

CHURN_DRIVERS_TITLE = "🔍 What Drives Customer Churn?"

CHURN_DRIVERS_INTRO = """
This analysis shows which factors the model relies on most when predicting churn.
These **do not imply causation**, but they highlight meaningful behavioral patterns.
"""

CHURN_DRIVERS_INSIGHTS = """
**Key Patterns Identified:**

- **Month-to-month contracts** are strongly linked to churn — commitment level matters.

- Customers with **low tenure** often leave before loyalty habits form.

- **Electronic check payment** correlates with churn, possibly signaling lower perceived trust or friction.

- **Higher monthly charges** increase churn risk when value is not clearly communicated.

- Customers **without online security or tech support** experience frustration and reduced stickiness.
"""

# =============================================================================
# SECTION 3: CHURN SEGMENTATION
# =============================================================================

CHURN_SEGMENTATION_TITLE = "🧩 Churn Patterns Across Customer Segments"

CHURN_SEGMENTATION_INTRO = """
Churn is **not evenly distributed**. Certain customer groups churn at much higher rates.
Use these visuals to identify **where retention budget should be focused**.
"""

# =============================================================================
# SECTION 4: CUSTOMER RISK LOOKUP
# =============================================================================

CUSTOMER_RISK_TITLE = "🎯 Individual Customer Churn Risk"

CUSTOMER_RISK_INTRO = """
Search for a customer to see:
- Their predicted churn risk
- The **top reasons** behind the risk score
- Suggested retention approach tailored to their situation
"""

# Risk level labels
RISK_HIGH = "HIGH RISK"
RISK_MEDIUM = "MEDIUM RISK"
RISK_LOW = "LOW RISK"

# Risk level descriptions
RISK_HIGH_DESC = "Urgent intervention recommended"
RISK_MEDIUM_DESC = "Proactive engagement needed"
RISK_LOW_DESC = "Standard service approach"

# Explanation text for risk classification
RISK_CLASSIFICATION_EXPLANATION = """
_This classification reflects how similar the customer is to past churners in key behaviors and account characteristics._
"""

# =============================================================================
# SECTION 5: RETENTION STRATEGY RECOMMENDATIONS
# =============================================================================

RETENTION_STRATEGY_TITLE = "✅ Turning Insight into Action"

RETENTION_STRATEGY_INTRO = """
Retention strategy should be aligned with **why** a customer is at risk.
The table below converts churn drivers into **targeted action recommendations**.
"""

RETENTION_STRATEGY_TABLE = """
| Churn Driver | Recommended Action |
|---|---|
| Month-to-month contract | Offer discounted upgrade to 6–12 month plan |
| Electronic check payment | Encourage autopay/credit card enrollment |
| Low tenure | Improve onboarding / welcome touchpoints |
| High monthly charges | Provide value justification or loyalty discount |
| High support issues | Route to priority support workflow |
| No online security | Bundle security services at discounted rate |
| No tech support | Offer complimentary tech support trial |
"""

# =============================================================================
# SECTION HEADERS & SUBHEADERS
# =============================================================================

FEATURE_IMPORTANCE_HEADER = "Global Feature Importance"
SHAP_EXPLANATION_HEADER = "SHAP Explanation for Individual Customer"
TOP_DRIVERS_HEADER = "Top Factors Increasing Churn Risk"
RECOMMENDED_ACTIONS_HEADER = "Recommended Actions"

# =============================================================================
# ACTION RECOMMENDATIONS BY RISK LEVEL
# =============================================================================

HIGH_RISK_ACTIONS = {
    "title": "🚨 High Risk Actions",
    "actions": [
        "Contact customer within 24 hours",
        "Offer premium retention incentive",
        "Assign dedicated account manager",
        "Provide contract upgrade with discount"
    ],
    "estimated_cost": "$100",
    "expected_roi": "1900%"
}

MEDIUM_RISK_ACTIONS = {
    "title": "⚠️ Medium Risk Actions",
    "actions": [
        "Send personalized retention offer",
        "Highlight unused service benefits",
        "Provide limited-time discount",
        "Schedule proactive check-in call"
    ],
    "estimated_cost": "$50",
    "success_rate": "65%"
}

LOW_RISK_ACTIONS = {
    "title": "✅ Low Risk Actions",
    "actions": [
        "Continue standard service",
        "Offer loyalty rewards program",
        "Cross-sell complementary services",
        "Send quarterly satisfaction survey"
    ],
    "estimated_cost": "$15",
    "success_rate": "85%"
}

# =============================================================================
# SEGMENTATION LABELS & DESCRIPTIONS
# =============================================================================

SEGMENT_LABELS = {
    "contract_type": {
        "Month-to-month": "High-risk flexible contracts",
        "One year": "Medium commitment customers",
        "Two year": "Lowest churn commitment"
    },
    "payment_method": {
        "Electronic check": "Highest churn payment method",
        "Credit card (automatic)": "Stable payment method",
        "Bank transfer (automatic)": "Stable payment method",
        "Mailed check": "Manual payment method"
    },
    "tenure_group": {
        "0-12 months": "New customers - highest risk",
        "13-24 months": "Early adopters",
        "25-48 months": "Established customers",
        "49+ months": "Long-term loyal customers"
    }
}

# =============================================================================
# FOOTER & METADATA
# =============================================================================

DASHBOARD_FOOTER = """
---
**Model Details:** Gradient Boosting Classifier | **Training Date:** Latest pipeline run
**For questions or feedback:** Contact the Analytics Team
"""

MODEL_DISCLAIMER = """
*Note: Predictions are probabilistic estimates based on historical patterns.
Use in conjunction with domain expertise and customer context.*
"""
