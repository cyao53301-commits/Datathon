"""
Random Forest with Gradient Boosting to Predict Airline Fares
Uses Market Competition, Distance,
Created by Team 50 on February 28-March 1, 2026 for the Datathon
"""

# ============================================================
# Econometric modelling of airline fares
# ============================================================

# -----------------------------
# Import essential libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay

import statsmodels.api as sm

import shap

# -----------------------------
# Load the (cleaned) dataset
# -----------------------------
df = pd.read_excel("/mnt/data/airline_ticket_dataset.xlsx")

# Sort chronologically (important for time-aware modeling)
df = df.sort_values(["Year", "quarter"])

# -----------------------------
# Engineer the essential features using a log transform to take into account right-tailed variability of pricing
# -----------------------------

# Log transforms (stabilizes variance)
df["log_fare"] = np.log(df["fare"])
df["log_distance"] = np.log(df["nsmiles"])
df["log_passengers"] = np.log(df["passengers"] + 1)

# ---- HHI Approximation ----
# We only have largest carrier share and low-fare share.
# So we approximate the remaining market share.

df["other_share"] = 1 - df["large_ms"] - df["lf_ms"]

# Make sure no negative shares (data safety)
df["other_share"] = df["other_share"].clip(lower=0)

df["HHI"] = (
        df["large_ms"]**2 +
        df["lf_ms"]**2 +
        df["other_share"]**2
)

# Nonlinear concentration term
df["HHI_sq"] = df["HHI"]**2

# ---- Hub Dynamics ----
# Use city passenger totals as a proxy for hub strength

df["hub_strength"] = np.log(
    df["TotalFaredPax_city1"] +
    df["TotalFaredPax_city2"]
)

# Strongest endpoint hub
df["hub_max"] = np.log(
    df[["TotalFaredPax_city1",
        "TotalFaredPax_city2"]].max(axis=1)
)

# ---- Seasonality ----
df["holiday_q4"] = (df["quarter"] == 4).astype(int)

# Year trend (post-COVID pricing normalization)
df["year_trend"] = df["Year"] - df["Year"].min()

# -----------------------------
# Time to define the features we believe contribute to all this
# -----------------------------
features = [
    "log_distance",
    "log_passengers",
    "HHI",
    "HHI_sq",
    "large_ms",
    "lf_ms",
    "hub_strength",
    "hub_max",
    "holiday_q4",
    "year_trend",
    "TotalPerLFMkts_city1",
    "TotalPerLFMkts_city2",
    "TotalPerPrem_city1",
    "TotalPerPrem_city2"
]

target = "log_fare"

# Time-aware split (train on pre-2025)
train = df[df["Year"] < 2025]
test = df[df["Year"] >= 2025]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# ============================================================
# Time to do some econometrics (Ordinary Least Squares with Inference)
# ============================================================

X_train_sm = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_sm).fit()

print("\n================ OLS RESULTS ================")
print(ols_model.summary())

# ============================================================
# Random Forest time
# ============================================================

rf = RandomForestRegressor(
    n_estimators=400,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("\nRandom Forest RMSE:",
      np.sqrt(mean_squared_error(y_test, rf_preds)))
print("Random Forest R2:",
      r2_score(y_test, rf_preds))

# ============================================================
# Gradient boost
# ============================================================

gb = GradientBoostingRegressor(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)

print("\nGradient Boosting RMSE:",
      np.sqrt(mean_squared_error(y_test, gb_preds)))
print("Gradient Boosting R2:",
      r2_score(y_test, gb_preds))

# ============================================================
# Highlight important features
# ============================================================

def plot_importance(model, title):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10,6))
    plt.title(title)
    plt.bar(range(len(indices)), importance[indices])
    plt.xticks(range(len(indices)),
               X_train.columns[indices],
               rotation=90)
    plt.tight_layout()
    plt.show()

plot_importance(rf, "Random Forest Feature Importance")
plot_importance(gb, "Gradient Boosting Feature Importance")

# ============================================================
# Interpretation using SHAP
# ============================================================

explainer = shap.Explainer(gb)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test)
shap.plots.bar(shap_values)

# ============================================================
# Partial Dependence Plots
# ============================================================

features_to_plot = ["HHI", "hub_max", "log_distance", "lf_ms"]

PartialDependenceDisplay.from_estimator(
    gb, X_train, features_to_plot
)
plt.show()

# ============================================================
# Counterfactual simulation
# ============================================================

def simulate_lcc_entry(row, new_lcc_share=0.25):
    row_cf = row.copy()

    row_cf["lf_ms"] = new_lcc_share
    row_cf["large_ms"] = max(0,
                             row["large_ms"] - new_lcc_share/2)

    other = 1 - row_cf["large_ms"] - row_cf["lf_ms"]
    other = max(0, other)

    row_cf["HHI"] = (
            row_cf["large_ms"]**2 +
            row_cf["lf_ms"]**2 +
            other**2
    )
    row_cf["HHI_sq"] = row_cf["HHI"]**2

    return row_cf

sample_route = X_test.iloc[0]

baseline_pred = gb.predict(sample_route.values.reshape(1, -1))

counterfactual = simulate_lcc_entry(sample_route)
cf_pred = gb.predict(counterfactual.values.reshape(1, -1))

print("\nCounterfactual Simulation")
print("Baseline fare:", np.exp(baseline_pred))
print("After LCC entry:", np.exp(cf_pred))
print("Percent change:",
      (np.exp(cf_pred) - np.exp(baseline_pred))
      / np.exp(baseline_pred))

# ============================================================
# Student-heavy route model for equity simulation
# ============================================================

# Proxy: high LCC penetration at both endpoints
df["student_proxy"] = (
        (df["TotalPerLFMkts_city1"]
         > df["TotalPerLFMkts_city1"].median()) &
        (df["TotalPerLFMkts_city2"]
         > df["TotalPerLFMkts_city2"].median())
).astype(int)

student_routes = df[df["student_proxy"] == 1]
non_student_routes = df[df["student_proxy"] == 0]

gb_student = GradientBoostingRegressor(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=3,
    random_state=42
)

gb_student.fit(
    student_routes[features],
    student_routes[target]
)

print("\nStudent Route Model R2:",
      gb_student.score(
          student_routes[features],
          student_routes[target]
      ))

# ============================================================
# The end
# ============================================================