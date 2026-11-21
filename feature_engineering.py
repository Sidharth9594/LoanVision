
import numpy as np


def add_features(X):
    """
    Create the four engineered features
    required by the trained loan default model.
    """

    X = X.copy()

    # 1. Calculate estimated monthly EMI
    r = X["interest_rate"] / 1200.0
    n = X["loan_term_months"]

    X["estimated_new_emi"] = (
        X["loan_amount"] * r * (1 + r) ** n
        / ((1 + r) ** n - 1)
    )

    # Calculate monthly income
    monthly_income = X["annual_income"] / 12.0

    # 2. Debt-to-Income ratio
    X["debt_to_income"] = (
        X["estimated_new_emi"] + X["existing_emi_monthly"]
    ) / monthly_income

    # 3. Loan-to-Income ratio
    X["loan_to_income"] = (
        X["loan_amount"] / X["annual_income"]
    )

    # 4. Savings-to-Income months
    X["savings_to_income_months"] = (
        X["savings_balance"] / monthly_income
    )

    # Replace infinite values with missing values
    return X.replace([np.inf, -np.inf], np.nan)
