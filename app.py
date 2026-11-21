
import os
import io
import uuid
from datetime import datetime, date

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from prediction import predict_loan_default


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LoanVision | Banking Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

HISTORY_FILE = "prediction_history.csv"
DATASET_FILE = "dataset_used.csv"

HISTORY_COLUMNS = [
    "applicant_id",
    "age",
    "education",
    "employment_status",
    "years_employed",
    "annual_income",
    "dependents",
    "home_ownership",
    "property_area",
    "loan_purpose",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "credit_score",
    "credit_history_years",
    "previous_loans",
    "late_payments_24m",
    "existing_emi_monthly",
    "savings_balance",
    "co_applicant",
    "collateral_provided",
    "prediction_date",
    "default_probability",
    "risk_category",
    "model_prediction",
    "application_status"
]



# ============================================================
# UI DESIGN
# ============================================================

st.markdown("""
<style>

/* MAIN APPLICATION */

.stApp {
    background: linear-gradient(
        135deg,
        #08111f 0%,
        #101c32 55%,
        #0b1628 100%
    );
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}

/* MAIN CONTENT */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #101d33,
        #0b1425
    );
    border-right: 1px solid #263b5a;
}

[data-testid="stSidebar"] h1 {
    color: #60a5fa;
    font-size: 27px;
    font-weight: 800;
}

[data-testid="stSidebar"] p {
    color: #a8b8cc;
}

/* HEADINGS */

h1 {
    color: #f8fafc !important;
    font-weight: 800 !important;
}

h2 {
    color: #93c5fd !important;
    font-weight: 750 !important;
}

h3 {
    color: #dbeafe !important;
    font-weight: 700 !important;
}

/* MAIN TITLE */

.main-title {
    font-size: 38px;
    font-weight: 850;
    letter-spacing: -1px;
    color: #60a5fa;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 15px;
    color: #94a3b8;
    letter-spacing: 0.3px;
}

/* SECTION TITLES */

.section-title {
    color: #60a5fa;
    font-size: 22px;
    font-weight: 750;
    margin-top: 15px;
}

/* METRIC CARDS */

div[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #192b46,
        #122139
    );
    border: 1px solid #29466b;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.18);
    transition: all 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: #60a5fa;
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(37,99,235,0.16);
}

div[data-testid="stMetricLabel"] {
    color: #a9bdd7 !important;
    font-size: 14px;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-weight: 800;
}

/* BUTTONS */

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(
        135deg,
        #2563eb,
        #3b82f6
    );
    color: white;
    border: 1px solid #60a5fa;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 700;
    transition: all 0.25s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(
        135deg,
        #1d4ed8,
        #2563eb
    );
    border-color: #93c5fd;
    transform: translateY(-2px);
}

/* INPUT FIELDS */

.stTextInput input,
.stNumberInput input,
.stDateInput input {
    background-color: #17263d !important;
    color: #f8fafc !important;
    border: 1px solid #36516f !important;
    border-radius: 9px !important;
}

/* SELECT BOX */

.stSelectbox > div > div {
    background-color: #17263d;
    border-radius: 9px;
}

/* FORM CONTAINERS */

[data-testid="stForm"] {
    background: rgba(23, 38, 61, 0.45);
    border: 1px solid #29466b;
    border-radius: 16px;
    padding: 22px;
}

/* DATA TABLE */

[data-testid="stDataFrame"] {
    border: 1px solid #29466b;
    border-radius: 12px;
    overflow: hidden;
}

/* CHART CONTAINERS */

[data-testid="stPlotlyChart"] {
    background: rgba(23, 38, 61, 0.45);
    border: 1px solid #29466b;
    border-radius: 15px;
    padding: 10px;
}

/* ALERT MESSAGES */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* DIVIDERS */

hr {
    border-color: #29405f;
}

/* FOOTER */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA FUNCTIONS
# ============================================================

def load_history():

    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    try:
        df = pd.read_csv(HISTORY_FILE)

        # Support older history files
        if "applicant_id" not in df.columns:
            df.insert(
                0,
                "applicant_id",
                ["LV-" + str(uuid.uuid4())[:8].upper()
                 for _ in range(len(df))]
            )

        if "application_status" not in df.columns:
            df["application_status"] = "Pending"

        for column in HISTORY_COLUMNS:
            if column not in df.columns:
                df[column] = np.nan

        return df[HISTORY_COLUMNS]

    except Exception as e:
        st.error(f"Unable to read prediction history: {e}")
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def save_history(applicant_data, result):

    history = load_history()

    record = applicant_data.copy()

    record["applicant_id"] = (
        "LV-" + str(uuid.uuid4())[:8].upper()
    )

    record["prediction_date"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    record["default_probability"] = result["probability"]
    record["risk_category"] = result["risk"]
    record["model_prediction"] = result["prediction"]
    record["application_status"] = "Pending"

    new_record = pd.DataFrame([record])

    history = pd.concat(
        [history, new_record],
        ignore_index=True
    )

    history = history[HISTORY_COLUMNS]

    history.to_csv(HISTORY_FILE, index=False)

    return record["applicant_id"]


def load_dataset():

    if os.path.exists(DATASET_FILE):
        return pd.read_csv(DATASET_FILE)

    return pd.DataFrame()


def format_currency(value):

    return f"₹{value:,.2f}"


def create_emi(principal, annual_rate, months):

    if months <= 0:
        return 0, 0, 0

    monthly_rate = annual_rate / (12 * 100)

    if monthly_rate == 0:
        emi = principal / months
    else:
        emi = (
            principal * monthly_rate *
            (1 + monthly_rate) ** months
            / ((1 + monthly_rate) ** months - 1)
        )

    total_payment = emi * months
    total_interest = total_payment - principal

    return emi, total_interest, total_payment


def create_pdf_report(record):

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle
        )
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet

        output = io.BytesIO()

        doc = SimpleDocTemplate(
            output,
            pagesize=A4
        )

        styles = getSampleStyleSheet()

        elements = []

        elements.append(
            Paragraph("LoanVision Prediction Report", styles["Title"])
        )

        elements.append(Spacer(1, 20))

        rows = [
            ["Field", "Value"],
            ["Applicant ID", str(record.get("applicant_id", ""))],
            ["Date", str(record.get("prediction_date", ""))],
            ["Age", str(record.get("age", ""))],
            ["Employment", str(record.get("employment_status", ""))],
            ["Education", str(record.get("education", ""))],
            ["Loan Purpose", str(record.get("loan_purpose", ""))],
            ["Loan Amount", format_currency(float(record["loan_amount"]))],
            ["Credit Score", str(record.get("credit_score", ""))],
            [
                "Default Probability",
                f"{float(record['default_probability']):.2%}"
            ],
            ["Risk Category", str(record.get("risk_category", ""))],
            [
                "Model Prediction",
                "Default" if int(record["model_prediction"]) == 1
                else "No Default"
            ],
            ["Application Status", str(record.get("application_status", ""))]
        ]

        table = Table(rows, colWidths=[170, 300])

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))

        elements.append(table)

        elements.append(Spacer(1, 25))

        elements.append(
            Paragraph(
                "Academic demonstration using a synthetic dataset. "
                "This report is not a real banking approval decision.",
                styles["Normal"]
            )
        )

        doc.build(elements)

        output.seek(0)

        return output.getvalue()

    except ImportError:
        return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏦 LoanVision")

    st.caption("BANKING INTELLIGENCE SYSTEM")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Loan Prediction",
            "EMI Calculator",
            "Applicant Management",
            "Prediction History",
            "Advanced Analytics",
            "Reports",
            "About Project"
        ]
    )

    st.divider()

    st.subheader("Project Details")

    st.write("**Project:** Bank Loan Default Prediction")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Technology:** Python and Streamlit")
    st.write("**Developer:** Sidharth Ilag")
    st.write("**Course:** M.Tech Computer Engineering")

    st.divider()

    st.caption("LoanVision | Academic Project | 2026")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏦 LoanVision</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Bank Loan Default Prediction and Risk Assessment System'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Dashboard Overview")

    df = load_dataset()
    history = load_history()

    if df.empty:
        st.warning("Dataset not found. Check dataset_used.csv.")
    else:

        total_applicants = len(df)

        default_count = int(df["loan_default"].sum())

        non_default_count = (
            total_applicants - default_count
        )

        default_rate = (
            df["loan_default"].mean() * 100
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Dataset Applicants",
            f"{total_applicants:,}"
        )

        col2.metric(
            "Loan Defaults",
            f"{default_count:,}"
        )

        col3.metric(
            "Non Defaults",
            f"{non_default_count:,}"
        )

        col4.metric(
            "Default Rate",
            f"{default_rate:.2f}%"
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average Loan",
            format_currency(df["loan_amount"].mean())
        )

        col2.metric(
            "Average Credit Score",
            f"{df['credit_score'].mean():.0f}"
        )

        col3.metric(
            "Saved Predictions",
            len(history)
        )

        st.divider()

        st.subheader("Loan Default Distribution")

        distribution = pd.DataFrame({
            "Status": ["No Default", "Default"],
            "Applicants": [
                non_default_count,
                default_count
            ]
        })

        fig = px.pie(
            distribution,
            names="Status",
            values="Applicants",
            hole=0.45,
            title="Default vs Non-Default"
        )

        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Loan Purpose")

            purpose = df["loan_purpose"].value_counts()

            st.bar_chart(purpose)

        with col2:

            st.subheader("Employment Status")

            employment = df["employment_status"].value_counts()

            st.bar_chart(employment)

        st.divider()

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


# ============================================================
# LOAN PREDICTION
# ============================================================

elif page == "Loan Prediction":

    st.header("🔍 Loan Default Prediction")

    st.write(
        "Enter applicant details to calculate default probability."
    )

    with st.form("applicant_prediction_form"):

        st.subheader("Personal Information")

        col1, col2, col3 = st.columns(3)

        age = col1.number_input(
            "Age", 21, 70, 37
        )

        education = col2.selectbox(
            "Education",
            ["Graduate", "Not Graduate", "Postgraduate"]
        )

        dependents = col3.number_input(
            "Dependents", 0, 4, 1
        )

        employment_status = st.selectbox(
            "Employment Status",
            [
                "Business Owner",
                "Retired",
                "Salaried",
                "Self-Employed"
            ]
        )

        years_employed = st.number_input(
            "Years Employed",
            0.0, 35.1, 6.2, step=0.1
        )

        st.divider()

        st.subheader("Loan Information")

        col1, col2, col3 = st.columns(3)

        loan_amount = col1.number_input(
            "Loan Amount (₹)",
            50000, 22715000, 700000,
            step=10000
        )

        loan_term_months = col2.number_input(
            "Loan Term (Months)",
            12, 240, 60, step=12
        )

        interest_rate = col3.number_input(
            "Interest Rate (%)",
            7.0, 17.82, 10.07, step=0.1
        )

        loan_purpose = st.selectbox(
            "Loan Purpose",
            ["Business", "Education", "Home", "Personal", "Vehicle"]
        )

        home_ownership = st.selectbox(
            "Home Ownership",
            ["Family", "Own", "Rent"]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Rural", "Semi-Urban", "Urban"]
        )

        st.divider()

        st.subheader("Financial Information")

        col1, col2 = st.columns(2)

        annual_income = col1.number_input(
            "Annual Income (₹)",
            120000, 3919000, 630000,
            step=10000
        )

        existing_emi_monthly = col2.number_input(
            "Existing Monthly EMI (₹)",
            0, 80700, 1200,
            step=500
        )

        col1, col2 = st.columns(2)

        savings_balance = col1.number_input(
            "Savings Balance (₹)",
            5000, 11610000, 221000,
            step=5000
        )

        late_payments_24m = col2.number_input(
            "Late Payments (Last 24 Months)",
            0, 4, 0
        )

        st.divider()

        st.subheader("Credit Information")

        col1, col2, col3 = st.columns(3)

        credit_score = col1.number_input(
            "Credit Score", 453, 900, 716
        )

        credit_history_years = col2.number_input(
            "Credit History (Years)",
            0.6, 32.0, 6.1, step=0.1
        )

        previous_loans = col3.number_input(
            "Previous Loans", 0, 8, 2
        )

        col1, col2 = st.columns(2)

        co_applicant = col1.selectbox(
            "Co-applicant", ["No", "Yes"]
        )

        collateral_provided = col2.selectbox(
            "Collateral", ["No", "Yes"]
        )

        submitted = st.form_submit_button(
            "Predict Loan Default Risk",
            use_container_width=True
        )

    if submitted:

        applicant_data = {
            "age": age,
            "education": education,
            "employment_status": employment_status,
            "years_employed": years_employed,
            "annual_income": annual_income,
            "dependents": dependents,
            "home_ownership": home_ownership,
            "property_area": property_area,
            "loan_purpose": loan_purpose,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "interest_rate": interest_rate,
            "credit_score": credit_score,
            "credit_history_years": credit_history_years,
            "previous_loans": previous_loans,
            "late_payments_24m": late_payments_24m,
            "existing_emi_monthly": existing_emi_monthly,
            "savings_balance": savings_balance,
            "co_applicant": 1 if co_applicant == "Yes" else 0,
            "collateral_provided": (
                1 if collateral_provided == "Yes" else 0
            )
        }

        try:

            result = predict_loan_default(applicant_data)

            applicant_id = save_history(
                applicant_data,
                result
            )

            st.session_state.latest_prediction = {
                **applicant_data,
                **result,
                "applicant_id": applicant_id
            }

            st.divider()

            st.header("Prediction Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Default Probability",
                f"{result['probability']:.2%}"
            )

            col2.metric(
                "Risk Category",
                result["risk"]
            )

            col3.metric(
                "Prediction",
                "Default" if result["prediction"] == 1
                else "No Default"
            )

            st.subheader("Probability Indicator")

            st.progress(
                float(result["probability"])
            )

            if result["risk"] == "Low":
                st.success("LOW RISK")
            elif result["risk"] == "Medium":
                st.warning("MEDIUM RISK")
            else:
                st.error("HIGH RISK")

            st.info(
                f"Applicant Reference ID: {applicant_id}"
            )

            st.caption(
                "Academic demonstration using a synthetic dataset. "
                "Predictions are estimates, not actual banking decisions."
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")


# ============================================================
# EMI CALCULATOR
# ============================================================

elif page == "EMI Calculator":

    st.header("🧮 EMI Calculator")

    st.write(
        "Calculate monthly EMI, total interest and repayment amount."
    )

    col1, col2 = st.columns(2)

    principal = col1.number_input(
        "Loan Amount (₹)",
        10000, 50000000, 700000,
        step=10000
    )

    rate = col2.number_input(
        "Annual Interest Rate (%)",
        0.0, 30.0, 10.0,
        step=0.1
    )

    months = st.number_input(
        "Loan Duration (Months)",
        1, 360, 60
    )

    emi, interest, total = create_emi(
        principal,
        rate,
        months
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Monthly EMI",
        format_currency(emi)
    )

    col2.metric(
        "Total Interest",
        format_currency(interest)
    )

    col3.metric(
        "Total Repayment",
        format_currency(total)
    )

    chart = pd.DataFrame({
        "Amount": ["Principal", "Interest"],
        "Value": [principal, interest]
    })

    fig = px.pie(
        chart,
        names="Amount",
        values="Value",
        hole=0.45,
        title="Loan Repayment Breakdown"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "EMI is an estimate based on a fixed interest rate. "
        "Actual lender charges may differ."
    )


# ============================================================
# APPLICANT MANAGEMENT
# ============================================================

elif page == "Applicant Management":

    st.header("👥 Applicant Management")

    history = load_history()

    if history.empty:

        st.info("No applicant records available yet.")

    else:

        st.metric(
            "Total Registered Applicants",
            len(history)
        )

        st.subheader("Search Applicant")

        applicant_search = st.text_input(
            "Enter Applicant ID"
        )

        if applicant_search:

            matching = history[
                history["applicant_id"].astype(str).str.contains(
                    applicant_search,
                    case=False,
                    na=False
                )
            ]

            st.dataframe(
                matching,
                use_container_width=True
            )

        else:

            st.dataframe(
                history,
                use_container_width=True
            )

        st.divider()

        st.subheader("Applicant Profile")

        selected_id = st.selectbox(
            "Select Applicant",
            history["applicant_id"].astype(str).tolist()
        )

        selected = history[
            history["applicant_id"].astype(str) == selected_id
        ]

        if not selected.empty:

            record = selected.iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Loan Amount",
                format_currency(float(record["loan_amount"]))
            )

            col2.metric(
                "Default Probability",
                f"{float(record['default_probability']):.2%}"
            )

            col3.metric(
                "Risk Category",
                str(record["risk_category"])
            )

            st.json(record.to_dict())

        st.divider()

        st.subheader("Update Application Status")

        status_id = st.selectbox(
            "Applicant Reference",
            history["applicant_id"].astype(str).tolist(),
            key="status_id"
        )

        new_status = st.selectbox(
            "New Status",
            ["Pending", "Under Review", "Approved", "Rejected"],
            key="new_status"
        )

        if st.button("Update Status"):

            history.loc[
                history["applicant_id"].astype(str) == status_id,
                "application_status"
            ] = new_status

            history.to_csv(HISTORY_FILE, index=False)

            st.success("Application status updated.")

            st.rerun()

        st.caption(
            "Application statuses are administrative labels for this "
            "academic demonstration, not actual lender decisions."
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "Prediction History":

    st.header("📜 Prediction History")

    history = load_history()

    if history.empty:

        st.info("No prediction history available.")

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Predictions",
            len(history)
        )

        col2.metric(
            "High Risk",
            int((history["risk_category"] == "High").sum())
        )

        col3.metric(
            "Medium Risk",
            int((history["risk_category"] == "Medium").sum())
        )

        st.divider()

        st.subheader("Search and Filter")

        col1, col2 = st.columns(2)

        search = col1.text_input(
            "Search Applicant ID, Education or Employment"
        )

        risk_filter = col2.selectbox(
            "Risk Category",
            ["All", "Low", "Medium", "High"]
        )

        filtered = history.copy()

        if search:

            columns = [
                "applicant_id",
                "education",
                "employment_status",
                "loan_purpose"
            ]

            mask = filtered[columns].astype(str).apply(
                lambda x: x.str.contains(
                    search,
                    case=False,
                    na=False
                )
            ).any(axis=1)

            filtered = filtered[mask]

        if risk_filter != "All":

            filtered = filtered[
                filtered["risk_category"] == risk_filter
            ]

        st.subheader("Date Filter")

        date_values = pd.to_datetime(
            history["prediction_date"],
            errors="coerce"
        ).dropna()

        if not date_values.empty:

            min_date = date_values.min().date()
            max_date = date_values.max().date()

            col1, col2 = st.columns(2)

            start_date = col1.date_input(
                "From Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )

            end_date = col2.date_input(
                "To Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )

            if start_date > end_date:
                st.error("Start date cannot be after end date.")
            else:

                filtered_dates = pd.to_datetime(
                    filtered["prediction_date"],
                    errors="coerce"
                )

                filtered = filtered[
                    filtered_dates.dt.date.between(
                        start_date,
                        end_date
                    )
                ]

        st.write(
            f"Showing {len(filtered)} records"
        )

        st.dataframe(
            filtered,
            use_container_width=True
        )

        st.divider()

        st.subheader("Download History")

        st.download_button(
            "Download CSV",
            data=history.to_csv(index=False),
            file_name="loan_prediction_history.csv",
            mime="text/csv"
        )

        st.download_button(
            "Download Excel",
            data=(
                lambda: (
                    lambda buffer: (
                        history.to_excel(buffer, index=False),
                        buffer.getvalue()
                    )[1]
                )(io.BytesIO())
            )(),
            file_name="loan_prediction_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.divider()

        st.subheader("Delete Individual Record")

        delete_id = st.selectbox(
            "Select Applicant ID to Delete",
            history["applicant_id"].astype(str).tolist()
        )

        if st.button("Delete Selected Record"):

            st.session_state.delete_record_id = delete_id
            st.session_state.confirm_record_delete = True

        if st.session_state.get("confirm_record_delete", False):

            st.warning(
                "Confirm deletion of this applicant record?"
            )

            col1, col2 = st.columns(2)

            if col1.button("Confirm Record Deletion"):

                updated = history[
                    history["applicant_id"].astype(str)
                    != st.session_state.delete_record_id
                ]

                updated.to_csv(HISTORY_FILE, index=False)

                st.session_state.confirm_record_delete = False

                st.success("Applicant record deleted.")

                st.rerun()

            if col2.button("Cancel Record Deletion"):

                st.session_state.confirm_record_delete = False

                st.rerun()

        st.divider()

        st.subheader("Delete Complete History")

        if st.button("🗑️ Delete All History"):

            st.session_state.confirm_all_delete = True

        if st.session_state.get("confirm_all_delete", False):

            st.error(
                "This permanently removes all prediction records."
            )

            col1, col2 = st.columns(2)

            if col1.button("Yes, Delete Everything"):

                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)

                st.session_state.confirm_all_delete = False

                st.success("All history deleted.")

                st.rerun()

            if col2.button("Cancel"):

                st.session_state.confirm_all_delete = False

                st.rerun()


# ============================================================
# ADVANCED ANALYTICS
# ============================================================

elif page == "Advanced Analytics":

    st.header("📈 Advanced Analytics")

    df = load_dataset()
    history = load_history()

    if df.empty:

        st.error("Dataset not found.")

    else:

        st.subheader("Loan Amount Analysis")

        fig = px.histogram(
            df,
            x="loan_amount",
            color="loan_default",
            nbins=30,
            title="Loan Amount Distribution by Default Status"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("Credit Score Analysis")

        fig = px.box(
            df,
            x="loan_default",
            y="credit_score",
            color="loan_default",
            title="Credit Score vs Loan Default"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("Employment Analysis")

        employment = pd.crosstab(
            df["employment_status"],
            df["loan_default"]
        )

        st.bar_chart(employment)

        st.divider()

        st.subheader("Loan Purpose Analysis")

        purpose = pd.crosstab(
            df["loan_purpose"],
            df["loan_default"]
        )

        st.bar_chart(purpose)

        st.divider()

        st.subheader("Prediction Risk Distribution")

        if not history.empty:

            risk_data = history["risk_category"].value_counts()

            fig = px.pie(
                values=risk_data.values,
                names=risk_data.index,
                hole=0.4,
                title="Saved Prediction Risk Categories"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Prediction Probability Distribution")

            fig = px.histogram(
                history,
                x="default_probability",
                color="risk_category",
                nbins=20
            )

            st.plotly_chart(fig, use_container_width=True)

        else:

            st.info(
                "Make some predictions to view prediction analytics."
            )


# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":

    st.header("📄 Reports and Exports")

    history = load_history()

    if history.empty:

        st.info("No prediction records available for reports.")

    else:

        st.subheader("Select Applicant")

        selected_id = st.selectbox(
            "Applicant Reference ID",
            history["applicant_id"].astype(str).tolist()
        )

        record = history[
            history["applicant_id"].astype(str) == selected_id
        ].iloc[0]

        st.write("### Selected Applicant Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Loan Amount",
            format_currency(float(record["loan_amount"]))
        )

        col2.metric(
            "Default Probability",
            f"{float(record['default_probability']):.2%}"
        )

        col3.metric(
            "Risk Category",
            str(record["risk_category"])
        )

        pdf_data = create_pdf_report(record)

        if pdf_data:

            st.download_button(
                "📥 Download Individual PDF Report",
                data=pdf_data,
                file_name=f"{selected_id}_report.pdf",
                mime="application/pdf"
            )

        else:

            st.warning(
                "PDF library not installed. Install it using: "
                "pip install reportlab"
            )

        st.divider()

        st.subheader("Complete Excel Report")

        excel_buffer = io.BytesIO()

        history.to_excel(
            excel_buffer,
            index=False
        )

        st.download_button(
            "Download All Applicant Records (Excel)",
            data=excel_buffer.getvalue(),
            file_name="LoanVision_Complete_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.divider()

        st.subheader("Prediction Summary")

        summary = history[
            [
                "applicant_id",
                "prediction_date",
                "loan_amount",
                "default_probability",
                "risk_category",
                "model_prediction"
            ]
        ]

        st.dataframe(
            summary,
            use_container_width=True
        )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.header("📘 About LoanVision")

    st.write("""
    LoanVision is a machine learning based Bank Loan Default
    Prediction and Risk Assessment System.

    It analyzes applicant financial information, credit history
    and loan details to estimate loan default probability.
    """)

    st.divider()

    st.subheader("Project Objectives")

    st.markdown("""
    - Analyze loan applicant information.
    - Estimate loan default probability.
    - Classify applicants into Low, Medium and High risk bands.
    - Calculate loan EMI and repayment details.
    - Maintain applicant prediction history.
    - Generate reports and analytical visualizations.
    - Support academic research into loan risk assessment.
    """)

    st.divider()

    st.subheader("Technology Stack")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Language:** Python")
        st.write("**Frontend:** Streamlit")
        st.write("**Data:** Pandas and NumPy")
        st.write("**Charts:** Plotly")

    with col2:
        st.write("**ML:** Scikit-learn")
        st.write("**Algorithm:** Logistic Regression")
        st.write("**Storage:** CSV")
        st.write("**Reports:** ReportLab and OpenPyXL")

    st.divider()

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Cross-Validated ROC-AUC",
        "0.7615"
    )

    col2.metric(
        "Test ROC-AUC",
        "0.7586"
    )

    col3.metric(
        "Test Accuracy",
        "74.13%"
    )

    st.info(
        "The model uses a synthetic academic dataset. "
        "Predictions are estimates and must not be used as "
        "actual lending or credit approval decisions."
    )

    st.divider()

    st.subheader("Developer Information")

    st.write("**Name:** Sidharth Ilag")
    st.write("**Department:** Computer Engineering")
    st.write("**Course:** M.Tech")
    st.write("**Project:** Bank Loan Default Prediction")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "LoanVision | Bank Loan Default Prediction | "
    "Developed by Sidharth Ilag | M.Tech Computer Engineering"
)
