#==============================================================
# Airfare market structure analysis by Team 50 - 2026/02/28
# Combines structural econometrics with XGBoost, Random Forest, Time Series, Counterfactual Policy Engine
#==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.inspection import PartialDependenceDisplay

from scipy.stats import randint, uniform
from xgboost import XGBRegressor
import shap
import statsmodels.api as sm

#==============================================================
# Load and Sort Data
#==============================================================

df = pd.read_excel("C:/Users/Soham Dhar/IdeaProjects/untitled/the project/airline_cleaned.xlsx")
df = df.sort_values(["Year", "quarter"])

#==============================================================
# Structural and Feature Engineering
#==============================================================

# Log transforms
df["log_fare"] = np.log(df["fare"])
df["log_distance"] = np.log(df["nsmiles"])
df["log_passengers"] = np.log(df["passengers"] + 1)

# Market shares
df["other_share"] = 1 - df["large_ms"] - df["lf_ms"]
df["other_share"] = df["other_share"].clip(lower=0)

# HHI (market concentration)
df["HHI"] = (
        df["large_ms"]**2 +
        df["lf_ms"]**2 +
        df["other_share"]**2
)

df["HHI_sq"] = df["HHI"]**2  # Nonlinear pricing power

# Hub strength
df["hub_strength"] = np.log(
    df["TotalFaredPax_city1"] + df["TotalFaredPax_city2"]
)

df["hub_max"] = np.log(
    df[["TotalFaredPax_city1",
        "TotalFaredPax_city2"]].max(axis=1)
)

df["hub_HHI_interaction"] = df["hub_max"] * df["HHI"]

# Seasonality + time trend
df["holiday_q4"] = (df["quarter"] == 4).astype(int)
df["year_trend"] = df["Year"] - df["Year"].min()

#==============================================================
# Structural segmentation
#==============================================================

# Hub routes (top quartile exposure)
hub_threshold = df["hub_max"].quantile(0.75)
df["hub_route"] = (df["hub_max"] >= hub_threshold).astype(int)

# Student-heavy markets proxy
df["student_proxy"] = (
        (df["TotalPerLFMkts_city1"] > df["TotalPerLFMkts_city1"].median()) &
        (df["TotalPerLFMkts_city2"] > df["TotalPerLFMkts_city2"].median())
).astype(int)

# Structural heterogeneity
df["student_HHI_interaction"] = df["student_proxy"] * df["HHI"]

#==============================================================
# Route mixed effects
#==============================================================

df["route_id"] = df["city1"].astype(str) + "_" + df["city2"].astype(str)
route_dummies = pd.get_dummies(df["route_id"], drop_first=True)
df = pd.concat([df, route_dummies], axis=1)

#==============================================================
# Competitive regime clustering (KMeans)
#==============================================================

strategy_vars = [
    "large_ms",
    "lf_ms",
    "TotalPerLFMkts_city1",
    "TotalPerLFMkts_city2",
    "TotalPerPrem_city1",
    "TotalPerPrem_city2"
]

scaler = StandardScaler()
strategy_scaled = scaler.fit_transform(df[strategy_vars])

kmeans = KMeans(n_clusters=3, random_state=42)
df["strategy_cluster"] = kmeans.fit_predict(strategy_scaled)

strategy_dummies = pd.get_dummies(
    df["strategy_cluster"],
    prefix="strategy",
    drop_first=True
)

df = pd.concat([df, strategy_dummies], axis=1)

#==============================================================
# Econometric baseline (OLS)
#==============================================================

ols_features = [
    "log_distance",
    "log_passengers",
    "HHI",
    "HHI_sq",
    "hub_max",
    "hub_HHI_interaction",
    "student_proxy",
    "student_HHI_interaction",
    "holiday_q4",
    "year_trend"
]

X_ols = sm.add_constant(df[ols_features])
y_ols = df["log_fare"]

ols_model = sm.OLS(y_ols, X_ols).fit()

print(ols_model.summary())

#==============================================================
# Feature set for ML
#==============================================================

base_features = ols_features
strategy_features = list(strategy_dummies.columns)
route_fe = list(route_dummies.columns)

features = base_features + strategy_features + route_fe
target = "log_fare"

#==============================================================
# Train/test split
#==============================================================

train = df[df["Year"] < 2025]
test = df[df["Year"] >= 2025]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

#==============================================================
# Random forest benchmark
#==============================================================

rf = RandomForestRegressor(
    n_estimators=800,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("RF RMSE:",
      np.sqrt(mean_squared_error(y_test, rf_preds)))
print("RF R2:",
      r2_score(y_test, rf_preds))

#==============================================================
# Time series tuned XGBoost
#==============================================================

tscv = TimeSeriesSplit(n_splits=5)

param_dist = {
    "n_estimators": randint(1000, 4000),
    "max_depth": randint(3, 6),
    "learning_rate": uniform(0.01, 0.05),
    "subsample": uniform(0.7, 0.3)
}

search = RandomizedSearchCV(
    XGBRegressor(objective="reg:squarederror"),
    param_distributions=param_dist,
    n_iter=20,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

xgb = search.best_estimator_

xgb_preds = xgb.predict(X_test)

print("XGB RMSE:",
      np.sqrt(mean_squared_error(y_test, xgb_preds)))
print("XGB R2:",
      r2_score(y_test, xgb_preds))

#==============================================================
# Shap interpretation
#==============================================================

explainer = shap.Explainer(xgb)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test)
shap.plots.bar(shap_values)

#==============================================================
# Partial dependence
#==============================================================

features_to_plot = ["HHI", "hub_max", "student_HHI_interaction"]

PartialDependenceDisplay.from_estimator(
    xgb,
    X_train,
    features_to_plot
)

plt.show()

#==============================================================
# Counterfactual merger engine
#==============================================================

df["predicted_fare"] = np.exp(
    xgb.predict(df[features])
)

def simulate_merger_full(data, transfer=0.15):

    df_cf = data.copy()

    df_cf["HHI_cf"] = df_cf["HHI"] * (1 + transfer)
    df_cf["HHI_sq_cf"] = df_cf["HHI_cf"]**2

    df_cf_temp = df_cf.copy()
    df_cf_temp["HHI"] = df_cf["HHI_cf"]
    df_cf_temp["HHI_sq"] = df_cf["HHI_sq_cf"]

    df_cf["fare_cf"] = np.exp(
        xgb.predict(df_cf_temp[features])
    )

    df_cf["percent_change"] = (
                                      df_cf["fare_cf"] - df["predicted_fare"]
                              ) / df["predicted_fare"]

    return df_cf

#==============================================================
# Segmented merger sensitivity
#==============================================================

transfer_grid = np.linspace(0.01, 0.40, 20)

hub_effects = []
nonhub_effects = []
student_effects = []
nonstudent_effects = []

for t in transfer_grid:

    df_cf = simulate_merger_full(df, transfer=t)

    hub_effects.append(
        df_cf[df_cf["hub_route"] == 1]["percent_change"].mean()
    )

    nonhub_effects.append(
        df_cf[df_cf["hub_route"] == 0]["percent_change"].mean()
    )

    student_effects.append(
        df_cf[df_cf["student_proxy"] == 1]["percent_change"].mean()
    )

    nonstudent_effects.append(
        df_cf[df_cf["student_proxy"] == 0]["percent_change"].mean()
    )

# Hub comparison
plt.plot(transfer_grid, hub_effects, label="Hub")
plt.plot(transfer_grid, nonhub_effects, label="Non-Hub")
plt.legend()
plt.title("Merger Sensitivity: Hub vs Non-Hub")
plt.show()

# Student comparison
plt.plot(transfer_grid, student_effects, label="Student-Heavy")
plt.plot(transfer_grid, nonstudent_effects, label="Other")
plt.legend()
plt.title("Merger Sensitivity: Student vs Non-Student")
plt.show()
