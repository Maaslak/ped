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
print(SELECT_FEATURES)
len(SELECT_FEATURES), len(train_df.columns)

# %%
train_df_selected = train_df[[feature for feature in SELECT_FEATURES if (not feature.endswith("_detected")) and feature not in ("has_detection", "face_count")]]
test_df_selected = test_df[[feature for feature in SELECT_FEATURES if (not feature.endswith("_detected")) and feature not in ("has_detection", "face_count")]]

# %% [markdown]
# ## Enocding categorical features

# %%
all_categories = np.concatenate([train_df.category_id.unique(), test_df.category_id.unique()])
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

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)

# %% [markdown]
# ### Explaining results

# %%
for i in range(5):
    display(shap.force_plot(explainer.expected_value, shap_values[i,:], X_train.iloc[i,:]))

# %%
shap_values_test = explainer.shap_values(X_test)
shap.force_plot(explainer.expected_value, shap_values_test, X_test)

# %%
from sklearn.metrics import accuracy_score

accuracy_score(y_test, model.predict(xgboost.DMatrix(X_test)) > 0.5)
