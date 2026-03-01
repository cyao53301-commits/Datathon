# ==============================================================
# Airline fare market structure Analysis
# Made by Team 50 in 2026/03/01
# Integrates Structural IO + ML + Counterfactual Policy Engine
# ==============================================================

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ML
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.inspection import PartialDependenceDisplay

from xgboost import XGBRegressor
import shap

# Econometrics
import statsmodels.api as sm
from scipy.stats import randint, uniform

# ==============================================================
# Load and sort data
# ==============================================================
df = pd.read_excel("C:/Users/Soham Dhar/IdeaProjects/untitled/the project/airline_cleaned.xlsx")
df = df.sort_values(["Year", "quarter"]).reset_index(drop=True)

# ==============================================================
# Feature engineering
# ==============================================================
df["log_fare"] = np.log(df["fare"])
df["log_distance"] = np.log(df["nsmiles"])
df["log_passengers"] = np.log(df["passengers"] + 1)

df["other_share"] = (1 - df["large_ms"] - df["lf_ms"]).clip(lower=0)
df["HHI"] = df["large_ms"]**2 + df["lf_ms"]**2 + df["other_share"]**2
df["HHI_sq"] = df["HHI"]**2
df["hub_max"] = np.log(df[["TotalFaredPax_city1","TotalFaredPax_city2"]].max(axis=1))
df["hub_HHI_interaction"] = df["hub_max"] * df["HHI"]

df["holiday_q4"] = (df["quarter"] == 4).astype(int)
df["year_trend"] = df["Year"] - df["Year"].min()

df["student_proxy"] = (
        (df["TotalPerLFMkts_city1"] > df["TotalPerLFMkts_city1"].median()) &
        (df["TotalPerLFMkts_city2"] > df["TotalPerLFMkts_city2"].median())
).astype(int)
df["student_HHI_interaction"] = df["student_proxy"] * df["HHI"]

hub_threshold = df["hub_max"].quantile(0.75)
df["hub_route"] = (df["hub_max"] >= hub_threshold).astype(int)

df["route_id"] = df["city1"].astype(str) + "_" + df["city2"].astype(str)
route_dummies = pd.get_dummies(df["route_id"], drop_first=True)
df = pd.concat([df, route_dummies], axis=1)

# ==============================================================
# Competitive regime clustering
# ==============================================================
strategy_vars = ["large_ms","lf_ms","TotalPerLFMkts_city1","TotalPerLFMkts_city2","TotalPerPrem_city1","TotalPerPrem_city2"]
scaler = StandardScaler()
strategy_scaled = scaler.fit_transform(df[strategy_vars])

kmeans = KMeans(n_clusters=3, random_state=42)
df["strategy_cluster"] = kmeans.fit_predict(strategy_scaled)
strategy_dummies = pd.get_dummies(df["strategy_cluster"], prefix="strategy", drop_first=True)
df = pd.concat([df, strategy_dummies], axis=1)

# ==============================================================
# Econometric baseline (OLS)
# ==============================================================
ols_features = [
    "log_distance","log_passengers","HHI","HHI_sq","hub_max","hub_HHI_interaction",
    "student_proxy","student_HHI_interaction","holiday_q4","year_trend"
]
X_ols = sm.add_constant(df[ols_features])
y_ols = df["log_fare"]
ols_model = sm.OLS(y_ols, X_ols).fit()
print("\n=== OLS SUMMARY ===")
print(ols_model.summary())

# ==============================================================
# Feature set for ML
# ==============================================================
features = ols_features + list(strategy_dummies.columns) + list(route_dummies.columns)
target = "log_fare"

# ==============================================================
# Train/test split
# ==============================================================
train = df[df["Year"] < 2025]
test = df[df["Year"] >= 2025]
X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# ==============================================================
# Random forest benchmark
# ==============================================================
rf = RandomForestRegressor(
    n_estimators=2000,
    max_depth=25,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("\n=== RANDOM FOREST PERFORMANCE ===")
print("RF RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))
print("RF R2:", r2_score(y_test, rf_preds))

# ROC curve for student/budget proxy
rf_probs = rf.predict(X_test)
fpr, tpr, thresholds = roc_curve(test["student_proxy"], rf_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='RF ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('Random Forest ROC Curve (Student Proxy)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# ==============================================================
# Time series XBoost with Randomized Search
# ==============================================================
# Time series CV
tscv = TimeSeriesSplit(n_splits=5)

# Hyperparameter distribution for policy-focused robustness
param_dist = {
    "n_estimators": randint(1000, 4000),
    "max_depth": randint(3, 6),
    "learning_rate": uniform(0.01, 0.03),  # slower learning
    "subsample": uniform(0.7, 0.3),
    "colsample_bytree": uniform(0.7, 0.3),
    "min_child_weight": randint(3, 6),
    "gamma": uniform(0.1, 0.3)
}

xgb_search = RandomizedSearchCV(
    XGBRegressor(objective="reg:squarederror", random_state=42),
    param_distributions=param_dist,
    n_iter=25,  # moderate search
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train, y_train)

xgb = xgb_search.best_estimator_

xgb_preds = xgb.predict(X_test)

print("XGB RMSE:", np.sqrt(mean_squared_error(y_test, xgb_preds)))
print("XGB R2:", r2_score(y_test, xgb_preds))
# ==============================================================
# Shap interpretability
# ==============================================================
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
shap.plots.bar(shap_values)

# ==============================================================
# Partial dependence
# ==============================================================
PartialDependenceDisplay.from_estimator(
    xgb,
    X_train,
    ["HHI","hub_max","student_HHI_interaction"]
)
plt.show()

# ==============================================================
# Counterfactual merger engine
# ==============================================================
df["baseline_pred_log"] = xgb.predict(df[features])
df["baseline_pred_fare"] = np.exp(df["baseline_pred_log"])

def simulate_merger(data, transfer=0.15):
    """Simulates HHI increase (merger) on predicted fares"""
    df_cf = data.copy()
    df_cf["HHI"] = df_cf["HHI"] * (1 + transfer)
    df_cf["HHI_sq"] = df_cf["HHI"]**2
    cf_log = xgb.predict(df_cf[features])
    df_cf["fare_cf"] = np.exp(cf_log)
    df_cf["percent_change"] = (df_cf["fare_cf"] - df_cf["baseline_pred_fare"]) / df_cf["baseline_pred_fare"]
    return df_cf

# ==============================================================
# Merger sensitivity by segment
# ==============================================================
transfer_grid = np.linspace(0.01,0.40,20)
hub_effects, nonhub_effects = [], []
student_effects, nonstudent_effects = [], []

for t in transfer_grid:
    df_cf = simulate_merger(df, transfer=t)
    hub_effects.append(df_cf[df_cf["hub_route"]==1]["percent_change"].mean())
    nonhub_effects.append(df_cf[df_cf["hub_route"]==0]["percent_change"].mean())
    student_effects.append(df_cf[df_cf["student_proxy"]==1]["percent_change"].mean())
    nonstudent_effects.append(df_cf[df_cf["student_proxy"]==0]["percent_change"].mean())

plt.plot(transfer_grid, hub_effects, label="Hub")
plt.plot(transfer_grid, nonhub_effects, label="Non-Hub")
plt.title("Merger Sensitivity: Hub vs Non-Hub")
plt.legend()
plt.show()

plt.plot(transfer_grid, student_effects, label="Student")
plt.plot(transfer_grid, nonstudent_effects, label="Non-Student")
plt.title("Merger Sensitivity: Student vs Non-Student")
plt.legend()
plt.show()

# ==============================================================
# Budget-constrained welfare optimization
# ==============================================================
elasticities = [-0.8, -1.0, -1.2, -1.5, -2.0]
for e in elasticities:
    df_cf = simulate_merger(df, transfer=0.15)
    welfare = (df_cf["percent_change"] * df["fare"] * df["passengers"]) / abs(e)
    print(f"Elasticity {e}: Total Welfare Change = {welfare.sum()}")

# ==============================================================
# Placebo test
# ==============================================================
df_placebo = df.copy()
df_placebo["fake_HHI"] = df_placebo["HHI"].shift(2)
placebo_model = sm.OLS(df_placebo["log_fare"], sm.add_constant(df_placebo["fake_HHI"].fillna(0))).fit()
print("\n=== PLACEBO TEST ===")
print(placebo_model.summary())

# ==============================================================
# Bootstrap uncertainty
# ==============================================================
boot_effects = []
for i in range(200):
    sample = df.sample(frac=1, replace=True)
    df_cf = simulate_merger(sample, transfer=0.15)
    boot_effects.append(df_cf["percent_change"].mean())

lower, upper = np.percentile(boot_effects, 5), np.percentile(boot_effects, 95)
print("\n95% Bootstrap CI for Fare Change:", lower, upper)