import sys

from sklearn.metrics import root_mean_squared_error

assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"

import numpy as np

import matplotlib as mpl

import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from CombinedAttributesAdder import CombinedAttributesAdder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, reciprocal
import matplotlib.pyplot as plt

def load_data():
    return pd.read_csv("datasets/housing/housing.csv")

housing = load_data()

# Create categorical column
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# Use stratified way of splitting data
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

# Get X train/test and y train/test
train_labels = strat_train_set["median_house_value"].copy()
test_labels  = strat_test_set["median_house_value"].copy()

X_train = strat_train_set.drop(columns=["median_house_value"])
X_test  = strat_test_set.drop(columns=["median_house_value"])

# Get the numerical/categorical columns
num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

def idx(c): 
    return num_features.index(c)

# Get column indices dynamically
col_names = ["total_rooms", "total_bedrooms", "population", "households"]
rooms_ix, bedrooms_ix, population_ix, households_ix = map(idx, col_names)

# Create full pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder(
        rooms_ix, bedrooms_ix, population_ix, households_ix,
        add_bedrooms_per_room=False
    )),
    ("scaler", StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
])

#############################################################################

# GRID SEARCH CV

# Have grid parameters for models: Linear Regression, Decision Tree Regression, 
# Random Forest Regression, and Support Vector Machine Regression

param_grid_linreg = [
    {"reg__fit_intercept": [True, False]}
]

param_grid_dt = [
    {"reg__criterion": ["squared_error", "friedman_mse", "absolute_error"],
     "reg__max_depth": [None, 5, 10, 20, 50],
     "reg__min_samples_split": [2, 5, 10, 20],
     "reg__min_samples_leaf": [1, 2, 5, 10],
     "reg__max_features": [None, "sqrt", "log2"]}
]

param_grid_rf = [
    {"reg__n_estimators": [50, 100, 200],
     "reg__criterion": ["squared_error", "absolute_error"],
     "reg__max_depth": [None, 10, 20, 50],
     "reg__min_samples_split": [2, 5, 10],
     "reg__min_samples_leaf": [1, 2, 5],
     "reg__max_features": [None, "sqrt", "log2"],
     "reg__bootstrap": [True, False]}
]

param_grid_svr = [
  {"reg__kernel": ["rbf"],
   "reg__C": [1, 3, 10, 30, 100],
   "reg__gamma": ["scale", 0.3, 0.1, 0.03, 0.01],
   "reg__epsilon": [0.05, 0.1, 0.2, 0.3]},
  {"reg__kernel": ["linear"],
   "reg__C": [1, 3, 10, 30, 100],
   "reg__epsilon": [0.05, 0.1, 0.2, 0.3]},
]

# Create models for each with random _state = 42

linreg_pipe = Pipeline([
    ("prep", full_pipeline),
    ("reg", LinearRegression())
])

dt_pipe = Pipeline([
    ("prep", full_pipeline),
    ("reg", DecisionTreeRegressor(random_state=42))
])

rf_pipe = Pipeline([
    ("prep", full_pipeline),
    ("reg", RandomForestRegressor(random_state=42))
])

svr_pipe = Pipeline([
    ("prep", full_pipeline),
    ("reg", SVR())
])

# Create object Grid Search CV (model, param_grid, cv=5, scoring='neg_mean_squared_error',
# return_train_score=True)

linreg_search = GridSearchCV(linreg_pipe, param_grid_linreg, scoring='neg_mean_squared_error',cv=5, 
                          return_train_score=True, n_jobs=-1)

dt_search = GridSearchCV(dt_pipe, param_grid_dt, cv=5, scoring='neg_mean_squared_error',
                          return_train_score=True, n_jobs=-1)

rf_search = GridSearchCV(rf_pipe, param_grid_rf, cv=5, scoring='neg_mean_squared_error',
                          return_train_score=True, n_jobs=-1)

svr_search = GridSearchCV(svr_pipe, param_grid_svr, scoring='neg_mean_squared_error',cv=5,
                          return_train_score=True, n_jobs=-1)

# Train object grid search cv: grid_search_cv.fit(X_train, train_labels)

linreg_search.fit(X_train, train_labels)
dt_search.fit(X_train, train_labels)
rf_search.fit(X_train, train_labels)
svr_search.fit(X_train, train_labels)

# Get the results
# Print best parameters for each model (best_estimator_)
# Could use grid_search.cv_results


# RANDOMIZED SEARCH CV

# Import
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint

# Use randint to create param_distribs

# Same parameter distribution for linear regression

rand_param_dist_dt = {
    "reg__criterion": ["squared_error", "friedman_mse", "absolute_error"],
    "reg__max_depth": randint(2, 51),           
    "reg__min_samples_split": randint(2, 21),   
    "reg__min_samples_leaf": randint(1, 21), 
    "reg__max_features": [None, "sqrt", "log2"],
}

rand_param_dist_rf = {
    "reg__n_estimators": randint(100, 801),
    "reg__criterion": ["squared_error", "absolute_error"],
    "reg__max_depth": randint(5, 51),
    "reg__min_samples_split": randint(2, 21),   
    "reg__min_samples_leaf": randint(1, 21),    
    "reg__max_features": [None, "sqrt", "log2"],
    "reg__bootstrap": [True, False],
}

rand_param_dist_svr = [
    {"reg__kernel": ["rbf"],
     "reg__C": reciprocal(1e0, 1e3),     # log-uniform in [1, 1000]
     "reg__gamma": reciprocal(1e-3, 1e0),# log-uniform in [1e-3, 1]
     "reg__epsilon": uniform(0.01, 0.39) # uniform in (0.01, 0.40)
    },
    {"reg__kernel": ["linear"],
     "reg__C": reciprocal(1e0, 1e3),
     "reg__epsilon": uniform(0.01, 0.39)}
]

# Create models for each with random _state = 42

rand_linreg_pipe = Pipeline([
    ("prep", full_pipeline),
    ("reg", LinearRegression(random_state=42))
])

rand_dt_pipe = Pipeline([
    ("prep", full_pipeline),
    ("reg", DecisionTreeRegressor(random_state=42))
])

rand_rf_pipe = Pipeline([
    ("prep", full_pipeline),
    ("reg", RandomForestRegressor(random_state=42))
])

rand_svr_pipe = Pipeline([
    ("prep", full_pipeline),
    ("reg", SVR(random_state=42))
])

# Create object Randomized Search CV (model, param_distributions=param_distribs,
# n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)

linreg_rand_search = RandomizedSearchCV(rand_linreg_pipe, param_distributions=param_grid_linreg,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)

dt_rand_search = RandomizedSearchCV(rand_dt_pipe, param_distributions=rand_param_dist_dt,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)

rf_rand_search = RandomizedSearchCV(rand_rf_pipe, param_distributions=rand_param_dist_rf,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)

svr_rand_search = RandomizedSearchCV(rand_svr_pipe, param_distributions=rand_param_dist_svr,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)

# Train object randomized search cv: random_search.fit(X_train, train_labels)

linreg_rand_search.fit(X_train, train_labels)
dt_rand_search.fit(X_train, train_labels)
rf_rand_search.fit(X_train, train_labels)
svr_rand_search.fit(X_train, train_labels)

# Get the results
# Print best parameters for each model (best_estimator_)
# Could use grid_search.cv_results


# COMPARE MODELS ON FINE TUNING PROCESS

results_summary = pd.DataFrame([
    {"Model": "Linear Regression", "Search": "Grid", "Best Score (CV RMSE)": np.sqrt(-linreg_search.best_score_), "Best Params": linreg_search.best_params_},
    {"Model": "Decision Tree",     "Search": "Grid", "Best Score (CV RMSE)": np.sqrt(-dt_search.best_score_),     "Best Params": dt_search.best_params_},
    {"Model": "Random Forest",     "Search": "Grid", "Best Score (CV RMSE)": np.sqrt(-rf_search.best_score_),     "Best Params": rf_search.best_params_},
    {"Model": "SVR",               "Search": "Grid", "Best Score (CV RMSE)": np.sqrt(-svr_search.best_score_),    "Best Params": svr_search.best_params_},

    {"Model": "Linear Regression", "Search": "Random", "Best Score (CV RMSE)": np.sqrt(-linreg_rand_search.best_score_), "Best Params": linreg_rand_search.best_params_},
    {"Model": "Decision Tree",     "Search": "Random", "Best Score (CV RMSE)": np.sqrt(-dt_rand_search.best_score_),     "Best Params": dt_rand_search.best_params_},
    {"Model": "Random Forest",     "Search": "Random", "Best Score (CV RMSE)": np.sqrt(-rf_rand_search.best_score_),     "Best Params": rf_rand_search.best_params_},
    {"Model": "SVR",               "Search": "Random", "Best Score (CV RMSE)": np.sqrt(-svr_rand_search.best_score_),    "Best Params": svr_rand_search.best_params_},
])

print(results_summary)


plt.figure(figsize=(10,6))
for search_type in ["Grid", "Random"]:
    subset = results_summary[results_summary["Search"] == search_type]
    plt.bar(subset["Model"] + " (" + search_type + ")", subset["Best Score (CV RMSE)"])

plt.ylabel("CV RMSE (lower is better)")
plt.xticks(rotation=45)
plt.title("Comparison of Models after Hyperparameter Tuning")
plt.show()


# CHOOSE BEST MODEL AMONG THE FOUR AND EVALUATE ON TEST SET

#best_model = rf_rand_search.best_estimator_
#final_predictions = best_model.predict(X_test)

#test_rmse = root_mean_squared_error(test_labels, final_predictions)
#print("Test RMSE:", test_rmse)

# Scatter plot: predictions vs true values
#plt.figure(figsize=(7,7))
#plt.scatter(test_labels, final_predictions, alpha=0.5)
#plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], "r--")
#plt.xlabel("True Values")
#plt.ylabel("Predictions")
#plt.title("Predicted vs True Values")
#plt.show()

# Residuals
#residuals = test_labels - final_predictions
#plt.figure(figsize=(10,6))
#plt.hist(residuals, bins=50)
#plt.xlabel("Prediction Error")
#plt.ylabel("Count")
#plt.title("Residuals Distribution")
#plt.show()

