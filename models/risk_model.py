import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def create_training_data():

    np.random.seed(42)

    records = []

    # Create synthetic historical project records
    # for prototype/model demonstration.

    for i in range(500):

        budget_used = np.random.uniform(20, 110)
        physical_progress = np.random.uniform(15, 100)

        financial_gap = (
            budget_used - physical_progress
        )

        schedule_delay = np.random.uniform(
            0, 24
        )

        milestone_delay = np.random.uniform(
            0, 60
        )

        # Synthetic risk label
        risk_value = (
            financial_gap * 0.35
            + schedule_delay * 1.5
            + milestone_delay * 0.4
        )

        if risk_value > 65:
            risk = 2       # HIGH

        elif risk_value > 35:
            risk = 1       # MEDIUM

        else:
            risk = 0       # LOW

        records.append([
            budget_used,
            physical_progress,
            financial_gap,
            schedule_delay,
            milestone_delay,
            risk
        ])

    return pd.DataFrame(
        records,
        columns=[
            "budget_used",
            "physical_progress",
            "financial_gap",
            "schedule_delay",
            "milestone_delay",
            "risk"
        ]
    )


def train_model():

    data = create_training_data()

    X = data[
        [
            "budget_used",
            "physical_progress",
            "financial_gap",
            "schedule_delay",
            "milestone_delay"
        ]
    ]

    y = data["risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, accuracy


def predict_project_risk(
    model,
    budget_used,
    physical_progress,
    financial_gap,
    schedule_delay,
    milestone_delay
):

    features = pd.DataFrame(
        [[
            budget_used,
            physical_progress,
            financial_gap,
            schedule_delay,
            milestone_delay
        ]],
        columns=[
            "budget_used",
            "physical_progress",
            "financial_gap",
            "schedule_delay",
            "milestone_delay"
        ]
    )

    probabilities = model.predict_proba(
        features
    )[0]

    prediction = model.predict(
        features
    )[0]

    # Convert class to human-readable risk
    risk_names = {
        0: "LOW",
        1: "MEDIUM",
        2: "HIGH"
    }

    risk_name = risk_names[prediction]

    risk_probability = (
        max(probabilities) * 100
    )

    return risk_name, risk_probability