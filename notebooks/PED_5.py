# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import pandas as pd
import numpy as np
import xgboost
import shap

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score

DATA_PATH = os.path.join("..", "data")

# %% [markdown]
# ## Reading data

# %%
train_df = pd.read_csv(os.path.join(DATA_PATH, "aggregated_train_no_embeddings.csv"), index_col=0)
test_df = pd.read_csv(os.path.join(DATA_PATH, "aggregated_test_no_embeddings.csv"), index_col=0)

# %%
list(train_df.columns)

# %%
import csv

# LOOKS LIKE WORST PYTHON FILE READING CODE :D (COPY PASTING IS EVEN WORSE xD)

categories = {}
with open(os.path.join('..', 'data', 'categories.csv')) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue
        else:
            categories[int(row[0])] = row[1]
        line_count += 1
        
    print(f'Processed {line_count} lines.')
    
categories

# %% [markdown]
# ## Selecting features

# %%
with open(os.path.join("..", "data", "anova_best_all_no_embeddings.json"), "r") as fp:
    ANOVA_BEST = json.load(fp)

with open(os.path.join("..", "data", "chi2_best_all_no_embeddings.json"), "r") as fp:
    CHI2_BEST = json.load(fp)

with open(os.path.join("..", "data", "mi_best_all_no_embeddings.json"), "r") as fp:
    MI_BEST = json.load(fp)

with open(os.path.join("..", "data", "rfecv_best_all_no_embeddings.json"), "r") as fp:
    RFECV_BEST = json.load(fp)

N = 20
SELECT_FEATURES = list(set([*ANOVA_BEST[:N], *CHI2_BEST[:N], *MI_BEST[:N], *RFECV_BEST[:N]]))
SELECT_FEATURES += ["category_id", "is_trending"]

BANNED_FEATURES = [
    'likes_median',
    'views_max',
    "views_median",
    'likes_max',
    'dislikes_median',
    'description_changes',
    'title_changes',
]

SELECT_FEATURES = [feature for feature in SELECT_FEATURES if feature not in BANNED_FEATURES]

print(SELECT_FEATURES)
len(SELECT_FEATURES), len(train_df.columns)

# %%
train_df_selected = train_df[[feature for feature in SELECT_FEATURES if (not feature.endswith("_detected")) and feature not in ("has_detection", "face_count")]]
test_df_selected = test_df[[feature for feature in SELECT_FEATURES if (not feature.endswith("_detected")) and feature not in ("has_detection", "face_count")]]

# %% [markdown]
# ## Enocding categorical features

# %%
all_categories = np.unique(np.concatenate([train_df.category_id.unique(), test_df.category_id.unique()]))
enc = OneHotEncoder()
enc.fit(all_categories.reshape(-1, 1))

def onehot_encode(df, columns=["category_id"]):
    for column in columns:
        encoded = enc.transform(df[column].values.reshape(-1, 1)).toarray().T
        for category_id, values in zip(enc.categories_[0], encoded):
            df[f"category_id_{int(category_id)}"] = values
        df = df.drop(columns=[column])
onehot_encode(train_df_selected)
onehot_encode(test_df_selected)

# %%
X_train = train_df_selected.loc[:, train_df_selected.columns != 'is_trending'].fillna(-1)
y_train = train_df_selected["is_trending"].values

X_test = test_df_selected.loc[:, test_df_selected.columns != 'is_trending'].fillna(-1)
y_test = test_df_selected["is_trending"].values

X_train.shape, y_train.shape

# %% [markdown]
# ## Training model

# %%
# load JS visualization code to notebook
shap.initjs()

# train XGBoost model
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X_train, label=y_train), 1000)

# %%
y_train_pred = model.predict(xgboost.DMatrix(X_train)) > 0.5

# %% [markdown]
# ## Test model on unseen data

# %%
from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_test, model.predict(xgboost.DMatrix(X_test)) > 0.5))

# %% [markdown]
# ## Explaining results

# %%
# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# %% [markdown]
# ### Global interpretation

# %%
shap.summary_plot(shap_values, X_train)

# %% [markdown]
# ### Local interpretation

# %%
for i in range(5):
    display(shap.force_plot(explainer.expected_value, shap_values[i,:], X_train.iloc[i,:]))

# %%
SAMPLE_SIZE = 200

sample_idxs = np.random.permutation(range(X_test.shape[0]))[:SAMPLE_SIZE]

shap_values_test = explainer.shap_values(X_test.iloc[sample_idxs])
shap.force_plot(explainer.expected_value, shap_values_test, X_test.iloc[sample_idxs])

# %% [markdown]
# ### Per category analysis

# %%
for category in all_categories:
    mask = (train_df_selected.category_id == category).values
    shap_values_category = shap_values[mask]
    X_train_category = X_train[mask]
    y_train_pred_category = y_train_pred[mask]
    y_train_category = y_train[mask]
    
    print(f"Category {categories[int(category)] if category in categories else 'Unknown'}\n"
        f"Number of samples {len(X_train_category)}\nAccuracy at those samples {accuracy_score(y_train_category, y_train_pred_category)}")
    
    display(shap.summary_plot(shap_values_category, X_train_category))
    
    if len(X_train_category) > SAMPLE_SIZE:
        sample_idxs = np.random.permutation(range(X_train_category.shape[0]))[:SAMPLE_SIZE]
        X_train_category = X_train_category.iloc[sample_idxs]
        shap_values_category = shap_values_category[sample_idxs]
    
    display(shap.force_plot(explainer.expected_value, shap_values_category, X_train_category))
