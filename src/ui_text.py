"""
UI Text Content for Customer Churn Prediction Dashboard

This module contains all narrative text, section headers, and business-focused
messaging used throughout the dashboard. Centralizing this content makes it
easier to maintain consistent tone and messaging.
"""

# =============================================================================
# PAGE TITLE & OVERVIEW
# =============================================================================

PAGE_TITLE = "Customer Churn Prediction Dashboard"

PAGE_OVERVIEW = """
This dashboard helps identify customers at high risk of churn, understand *why*,
and guide data-driven retention strategy.
"""

# =============================================================================
# SECTION 1: BUSINESS CONTEXT
# =============================================================================

BUSINESS_CONTEXT_TITLE = "💡 Project Summary — Click to Expand"

BUSINESS_CONTEXT_CONTENT = """
Customer churn significantly impacts recurring revenue and growth stability.
This model predicts which customers are most likely to churn and highlights the
key behavioral and account factors driving that risk.

**Key Objectives**
- Identify high-risk customers
- Understand churn drivers at the feature level
- Support targeted retention strategies
"""

# =============================================================================
# SECTION 2: CHURN DRIVER INSIGHTS (SHAP)
# =============================================================================

CHURN_DRIVERS_TITLE = "🔍 Key Factors Driving Churn"

CHURN_DRIVERS_INTRO = """
These feature importance results explain what the model is learning and why
certain customers churn.
"""

CHURN_DRIVERS_INSIGHTS = """
**Key Takeaways from Feature Analysis:**
- **Month-to-month contracts** correlate heavily with churn
- **Low tenure customers** are at highest risk early in lifecycle
- **Electronic check payment** is a significant churn indicator
- **Higher monthly charges** increase churn likelihood
- **Lack of online security** services correlates with higher churn risk
"""

# =============================================================================
# SECTION 3: CHURN SEGMENTATION
# =============================================================================

CHURN_SEGMENTATION_TITLE = "🧩 Churn Segmentation by Customer Attributes"

CHURN_SEGMENTATION_INTRO = """
This view highlights how churn varies across different customer segments.
Use this to identify groups where targeted interventions will be most effective.
"""

# =============================================================================
# SECTION 4: CUSTOMER RISK LOOKUP
# =============================================================================

CUSTOMER_RISK_TITLE = "🎯 Customer-Specific Churn Risk"

CUSTOMER_RISK_INTRO = """
Enter a customer ID to see personalized churn risk and explanation.
This helps support teams tailor retention strategies at the individual level.
"""

# Risk level labels
RISK_HIGH = "HIGH RISK"
RISK_MEDIUM = "MEDIUM RISK"
RISK_LOW = "LOW RISK"

# Risk level descriptions
RISK_HIGH_DESC = "Urgent intervention recommended"
RISK_MEDIUM_DESC = "Proactive engagement needed"
RISK_LOW_DESC = "Standard service approach"

# =============================================================================
# SECTION 5: RETENTION STRATEGY RECOMMENDATIONS
# =============================================================================

RETENTION_STRATEGY_TITLE = "✅ Recommended Retention Strategies"

RETENTION_STRATEGY_INTRO = """
Use these data-driven recommendations to design targeted retention campaigns
based on the key churn drivers identified by the model.
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
