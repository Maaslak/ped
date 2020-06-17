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

# %% [markdown]
# ## Read Aggregated Data

# %%
import pandas as pd
import numpy as np
import os

agg_df = pd.read_csv('../data/aggregated.csv')
print(agg_df.shape)
agg_df.columns

# %% [markdown]
# ## Read simple category_id -> title mapper

# %%
import csv

# LOOKS LIKE WORST PYTHON FILE READING CODE :D

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
# ### Apply PCA over those multi-one-hot vectors

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(18, 14)})

title_onehot_feature_columns = list(filter(lambda x : 'title' in x and 'bin' in x, agg_df.columns))
X = agg_df[title_onehot_feature_columns].values
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_df["category_id"].fillna(0).values)
plt.show()

# %%
category_id_indices = agg_df.index[~agg_df["category_id"].isna()].tolist()
plt.scatter(X_pca[category_id_indices, 0], X_pca[category_id_indices, 1], c=agg_df.loc[category_id_indices, "category_id"])
plt.show()

# %% [markdown]
# ## Apply PCA over all columns, normalized by mean and std

# %%

agg_df_embeddings = agg_df[[cname for cname in agg_df.columns if cname.startswith('embed_')]]

def cast_to_list(x):
    if x:
        return [float(num) for num in x[1:-1].replace("\n", "").split(", ") if num]
    else:
        return None


for column in agg_df_embeddings.columns:
    agg_df_embeddings[column] = agg_df_embeddings[column].apply(cast_to_list)

agg_df_embeddings_numeric = pd.concat([
    pd.DataFrame(agg_df_embeddings[colname].values.tolist()).add_prefix(colname + '_')
    for colname in agg_df_embeddings.columns
], axis=1)

# %%

len(agg_df_embeddings_numeric.columns)

# %%
agg_df_numeric = agg_df[[cname for idx, cname in enumerate(agg_df.columns) if agg_df.dtypes[idx] in [np.int64, np.float64]]]
agg_df_not_numeric = agg_df[[cname for idx, cname in enumerate(agg_df.columns) if agg_df.dtypes[idx] not in [np.int64, np.float64]]]
agg_df_embeddings = agg_df[[cname for cname in agg_df.columns if cname.startswith('embed_')]]

agg_df_numeric = pd.concat([agg_df_numeric, agg_df_embeddings_numeric], axis=1)

all_numeric_df = agg_df_numeric.reset_index().fillna(-1).drop(columns=['trending_date', 'category_id'])
normalized_df = (all_numeric_df - all_numeric_df.mean()) / all_numeric_df.std()

X = normalized_df.values
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
ax = plt.gca()
plt.show()

# %% [markdown]
# ## Select features based on previous checkpoint's analysis

# %%
import json

with open(os.path.join("..", "data", "anova_best.json"), "r") as fp:
    ANOVA_BEST = json.load(fp)

with open(os.path.join("..", "data", "chi2_best.json"), "r") as fp:
    CHI2_BEST = json.load(fp)

with open(os.path.join("..", "data", "mi_best.json"), "r") as fp:
    MI_BEST = json.load(fp)

with open(os.path.join("..", "data", "rfecv_best.json"), "r") as fp:
    RFECV_BEST = json.load(fp)

N = 20
SELECT_FEATURES = list(set([*ANOVA_BEST[:N], *CHI2_BEST[:N], *MI_BEST[:N], *RFECV_BEST[:N]]))
len(SELECT_FEATURES), len(agg_df.columns)

# %% [markdown]
# ## Apply PCA over SELECTED FEATURES

# %%
select_features_df = agg_df_numeric.fillna(0)[SELECT_FEATURES]
normalized_df = (select_features_df - select_features_df.mean()) / select_features_df.std()

X_all = normalized_df.values
y_all = list(map(int, agg_df.fillna(-1).loc[:, "category_id"].values))

pca_all = PCA(n_components=5)
X_pca_all = pca_all.fit_transform(X_all)

import seaborn as sns

sns.scatterplot(
    x='c1', 
    y='c2',
      hue='category',
    size='has_category',
    data=pd.DataFrame({
      'c1': X_pca_all[:, 0],
      'c2': X_pca_all[:, 1],
      'category': list(map(lambda x : categories.get(int(x), "undefined"), y_all)),
        'has_category': list(map(lambda x : 1 if x == -1 else 15, y_all))
  }))
plt.show()

# %%
labeled_idx = agg_df.index[~agg_df["category_id"].isna()].tolist()
X = normalized_df.loc[labeled_idx, :].values
y = list(map(int, agg_df.loc[labeled_idx, "category_id"].values))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

import seaborn as sns

sns.scatterplot(
    x='c1', 
    y='c2',
      hue='category',
    data=pd.DataFrame({
      'c1': X_pca[:, 0],
      'c2': X_pca[:, 1],
      'category': list(map(lambda x : categories.get(int(x), "undefined"), y)),
  }))

plt.show()

# %%
_ = normalized_df.hist(bins=20)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Distribution of known categories

# %%
ax = sns.countplot(
    x="category", 
    data=pd.DataFrame({"category": map(lambda x : categories.get(x),filter(lambda x : x > -1, y_all))})
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Try: supervised apprroach vs. naive Self Learning Model

# %%
import numpy as np
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from frameworks.SelfLearning import SelfLearningModel

# supervised score 
# basemodel = WQDA() # weighted Quadratic Discriminant Analysis
basemodel = SGDClassifier(loss='log', penalty='l2', random_state=20200501)  # scikit logistic regression
basemodel.fit(X, y)
print("supervised log.reg. score", basemodel.score(X, y))  # 0.8426395939086294

y = np.array(y)
y_all = np.array(y_all)

# # fast (but naive, unsafe) self learning framework
ssmodel = SelfLearningModel(basemodel)
ssmodel.fit(X_all, y_all)
print("self-learning log.reg. score", ssmodel.score(X, y))  # 0.25380710659898476

# %% [markdown]
# ## Label Spreading

# %%
from sklearn.semi_supervised import LabelSpreading

# label_spread = LabelSpreading(kernel='knn', alpha=0.8, max_iter=1000)
label_spread = LabelSpreading(kernel='knn', alpha=0.2, max_iter=1000)

label_spread.fit(X_all, y_all)

# %%
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report

y_pred = label_spread.predict(X)
cm = confusion_matrix(y, y_pred, labels=label_spread.classes_)

print(classification_report(y, y_pred))

disp = plot_confusion_matrix(label_spread, X, y,
                                 display_labels=label_spread.classes_,
                                 cmap=plt.cm.Blues)

#               precision    recall  f1-score   support
#            1       0.38      0.75      0.51        20
#            2       0.00      0.00      0.00         3
#           10       0.79      0.91      0.84        54
#           15       1.00      0.60      0.75         5
#           17       0.88      0.76      0.81        29
#           19       0.00      0.00      0.00         2
#           20       0.92      0.79      0.85        14
#           22       0.89      0.82      0.85        39
#           23       0.86      0.78      0.82        40
#           24       0.90      0.86      0.88       100
#           25       0.91      0.83      0.87        24
#           26       0.81      0.94      0.87        32
#           27       0.86      0.60      0.71        10
#           28       0.89      0.76      0.82        21
#           29       1.00      1.00      1.00         1
#     accuracy                           0.82       394
#    macro avg       0.74      0.69      0.71       394
# weighted avg       0.83      0.82      0.82       394


# %%
label_spread_pred_all = label_spread.predict(X_all)

sns.scatterplot(
    x='c1', 
    y='c2',
      hue='category',
    size='correct',
    data=pd.DataFrame({
      'c1': X_pca_all[:, 0],
      'c2': X_pca_all[:, 1],
      'category': list(map(lambda x: categories.get(int(x), "undefined"),
                          label_spread_pred_all)),
        'correct': list(map(
            lambda x : 15 if x[0] == x[1] else 1, zip(y_all, label_spread_pred_all)))
  }))

plt.show()

# %%

correct_labels_df = pd.read_csv(os.path.join("..", "data", "labels_trending"))
mask_no_missing = correct_labels_df.category_id_true != -1

def plot_validation(y_true_series, y_pred):
    print(classification_report(y_true_series.values, y_pred))

    classes = sorted(y_true_series.unique())

    cm = confusion_matrix(y_true_series.values, y_pred, labels=classes)
    class_labels = [categories[int_class] for int_class in classes]
    sns.heatmap(cm, annot=True, yticklabels=class_labels, xticklabels=class_labels)
    plt.show()

plot_validation(correct_labels_df[mask_no_missing].category_id_true, label_spread_pred_all[mask_no_missing.values])

# %% [markdown]
# ## Entropies

# %%
from scipy import stats

# #############################################################################
# Calculate uncertainty values for each transduced distribution
pred_entropies = stats.distributions.entropy(label_spread.label_distributions_.T)
print(pred_entropies.shape)

sns.distplot(pred_entropies)
plt.show()

# %% [markdown]
# ### Read original dataframe to reference original titles & tags

# %%
path = "../data/"

GB_videos_df = pd.read_csv(path + "/" + "GB_videos_5p.csv", sep=";", engine="python")
US_videos_df = pd.read_csv(path + "/" + "US_videos_5p.csv", sep=";", engine="python")

df = pd.concat([GB_videos_df, US_videos_df]).drop_duplicates().reset_index(drop=True)
df = df.rename(columns={"description ": "description"})
print(df.shape)
df.head(3) 

# %% [markdown]
# ## Least certain

# %%
transductions_entropies = list(zip(
    label_spread.transduction_, 
    pred_entropies,
    [i for i in range(len(pred_entropies))]
))

for c in label_spread.classes_:
    print("\nCATEGORY", categories.get(c))
    print(">>> SUPPORT: ", len(list(filter(lambda x : x == c, y_all))), "\n")
    
    t_e_per_class = list(filter(lambda x : x[0] == c, transductions_entropies))
    t_e_per_class = list(sorted(t_e_per_class, key=lambda x : -1*x[1]))
    for _, entropy, idx in t_e_per_class[:5]:
        print(entropy)
        vid_id = agg_df.loc[idx, ["video_id"]].values[0]
        select_from_df = df[df["video_id"] == vid_id]
        print(select_from_df.loc[:, ["title"]].values[0][0])
        print(select_from_df.loc[:, ["tags"]].values[0][0])
        print()


# %% [markdown]
# ## Most certain

# %%
transductions_entropies = list(zip(
    label_spread.transduction_, 
    pred_entropies,
    [i for i in range(len(pred_entropies))]
))

for c in label_spread.classes_:
    print("\nCATEGORY", categories.get(c))
    print(">>> SUPPORT: ", len(list(filter(lambda x : x == c, y_all))), "\n")
    
    t_e_per_class = list(filter(lambda x : x[0] == c, transductions_entropies))
    t_e_per_class = list(sorted(t_e_per_class, key=lambda x : x[1]))
    for _, entropy, idx in t_e_per_class[:5]:
        print(entropy)
        vid_id = agg_df.loc[idx, ["video_id"]].values[0]
        select_from_df = df[df["video_id"] == vid_id]
        if select_from_df.shape[0] > 0:
            print(select_from_df.loc[:, ["title"]].values[0][0])
            print(select_from_df.loc[:, ["tags"]].values[0][0][:100])
            print()
# %% [markdown]
# # 2 Method: Gaussian Mixture Model
#
# First (bad) implementation found at kaggle site

# %%

# import numpy as np
# from scipy import stats


# class SSGaussianMixture(object):
#     def __init__(self, n_features, n_categories):
#         self.n_features = n_features
#         self.n_categories = n_categories

#         self.mus = np.array([np.random.randn(n_features)] * n_categories)
#         self.sigmas = np.array([np.eye(n_features)] * n_categories)
#         self.pis = np.array([1 / n_categories] * n_categories)

#     def fit(self, X_train, y_train, X_test, threshold=0.00001, max_iter=100):
#         Z_train = np.eye(self.n_categories)[y_train]

#         for i in range(max_iter):
#             # EM algorithm
#             # M step
#             Z_test = np.array([self.gamma(X_test, k) for k in range(self.n_categories)]).T
#             Z_test /= Z_test.sum(axis=1, keepdims=True)

#             # E step
#             datas = [X_train, Z_train, X_test, Z_test]
#             mus = np.array([self._est_mu(k, *datas) for k in range(self.n_categories)])
#             sigmas = np.array([self._est_sigma(k, *datas) for k in range(self.n_categories)])
#             pis = np.array([self._est_pi(k, *datas) for k in range(self.n_categories)])

#             diff = max(np.max(np.abs(mus - self.mus)),
#                        np.max(np.abs(sigmas - self.sigmas)),
#                        np.max(np.abs(pis - self.pis)))

#             print(f"{i + 1}/{max_iter} diff = {diff} conv matrix max = {np.max(sigmas)} min {np.min(sigmas)}")
#             self.mus = mus
#             self.sigmas = sigmas
#             self.pis = pis
#             if diff < threshold:
#                 break

#     def predict_proba(self, X):
#         Z_pred = np.array([self.gamma(X, k) for k in range(self.n_categories)]).T
#         Z_pred /= Z_pred.sum(axis=1, keepdims=True)
#         return Z_pred

#     def gamma(self, X, k):
#         # X is input vectors, k is feature index
#         return stats.multivariate_normal.pdf(X, mean=self.mus[k], cov=self.sigmas[k], allow_singular=True)

#     def _est_mu(self, k, X_train, Z_train, X_test, Z_test):
#         mu = (Z_train[:, k] @ X_train + Z_test[:, k] @ X_test).T / \
#              (Z_train[:, k].sum() + Z_test[:, k].sum())
#         return mu

#     def _est_sigma(self, k, X_train, Z_train, X_test, Z_test):
#         cmp1 = (X_train - self.mus[k]).T @ np.diag(Z_train[:, k]) @ (X_train - self.mus[k])
#         cmp2 = (X_test - self.mus[k]).T @ np.diag(Z_test[:, k]) @ (X_test - self.mus[k])
#         sigma = (cmp1 + cmp2) / (Z_train[:, k].sum() + Z_test[:k].sum())
#         return sigma

#     def _est_pi(self, k, X_train, Z_train, X_test, Z_test):
#         pi = (Z_train[:, k].sum() + Z_test[:, k].sum()) / \
#              (Z_train.sum() + Z_test.sum())
#         return pi

# # Below is just a lapper object.

# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

# from sklearn import preprocessing


# class BaseClassifier(object):
#     def __init__(self, n_categories):
#         self.n_categories = n_categories
#         self.preprocess = Pipeline([('scaler', StandardScaler())])
#         self.label_encoder = preprocessing.LabelEncoder()

#     def fit(self, X_train, y_train, X_test, max_iter=10, cv_qda=2, cv_meta=2):
#         X_train_org = X_train
#         self.label_encoder.fit(y_train)
#         y_train = self.label_encoder.transform(y_train)

#         self.preprocess_tune(np.vstack([X_train, X_test]))
#         X_train = self.preprocess.transform(X_train)
#         X_test = self.preprocess.transform(X_test)

#         self.cgm = SSGaussianMixture(
#             n_features=X_train.shape[1],
#             n_categories=self.n_categories,
#         )
#         _, unique_counts = np.unique(y, return_counts=True)
#         self.cgm.pis = unique_counts / np.sum(unique_counts)
#         self.cgm.fit(X_train, y_train, X_test, max_iter=max_iter)

#     def predict(self, X):
#         X = self.preprocess.transform(X)
#         y_prob = self.cgm.predict_proba(X)
#         y = np.argmax(y_prob, axis=-1)
#         return self.label_encoder.inverse_transform(y)

#     def preprocess_tune(self, X):
#         self.preprocess.fit(X)

#     def validation(self, X, y):
#         y_pred = self.predict(X)

#         cm = confusion_matrix(y, y_pred)  # , labels=label_spread.classes_)

#         print(classification_report(y, y_pred))

#         sns.heatmap(cm, annot=True)
#         plt.show()

# n_categoties = len(np.unique(y))
# bc = BaseClassifier(n_categoties)


# %% [markdown]
# ### Findig correlated embeddings features

# %%
corr_mat = pd.DataFrame(X_all).corr()
plt.matshow(corr_mat)

# %%
np.max(np.max(corr_mat[corr_mat != 1])), np.min(np.min(corr_mat))

# %%
np.array([
    pair
    for pair in 
    np.concatenate([np.array(np.where(np.logical_and(corr_mat > 0.5, corr_mat < 1.0))).T, np.array(np.where(corr_mat < -0.5)).T])
    if pair[0] < pair[1]
])

# %% [markdown]
# Removing corelated features

# %%
# Decided to remove
to_be_removed = [5, 12, 41, 58, 6, 13, 33, 46, 50]
cleaned_X_all = np.delete(X_all, to_be_removed, axis=1)
cleaned_X = np.delete(X, to_be_removed, axis=1)

corr_mat = pd.DataFrame(cleaned_X_all).corr()
plt.matshow(corr_mat)

# %%
np.max(np.max(corr_mat[corr_mat != 1])), np.min(np.min(corr_mat))

# %%
cleaned_X_no_labels = cleaned_X_all[y_all == -1]

# %%
np.unique(cleaned_X_no_labels, axis=0).shape, cleaned_X_no_labels.shape

# %%
cleaned_X_no_labels.shape, cleaned_X_no_labels[:,:20].shape

# %% [markdown]
# ### First approach generating very poor results

# %%
# bc.fit(cleaned_X, y, cleaned_X_no_labels, max_iter=20)
# bc.validation(cleaned_X, y)

# %% [markdown]
# ### Our implementation of SSGMM

# %%
unique_labels = list(np.unique(y))


# %%

from scipy.stats import multivariate_normal
import bidict
label_mapping = bidict.bidict({
    label_original: label_encoded
    for label_original, label_encoded in zip(unique_labels + [-1], list(range(len(unique_labels))) + [-1])
})


def get_probs_ssgmm(X, y, num_iterations=5):
    y = np.array([
        label_mapping[sing_y]
        for sing_y in y
    ])
    num_samples, n_features = X.shape
    unique_labels, unique_counts = np.unique(y, return_counts=True)
    unique_counts = unique_counts[unique_labels != -1]
    n_categories = len(unique_labels) - 1  # there is additional -1 label

    means = np.array([np.random.randn(n_features)] * n_categories)
    covs = np.array([np.eye(n_features)] * n_categories)
    qs = unique_counts / np.sum(unique_counts)

    print(means.shape)

    for iters in range(num_iterations):
        Pij = np.zeros((num_samples, n_categories))
        for i in range(num_samples):
            if y[i] == -1:
                ps = np.array([
                    multivariate_normal.pdf(X[i], means[cat_num], covs[cat_num], allow_singular=True) * q
                    for cat_num, q in zip(range(n_categories), qs)
                ])
                Pij[i] = ps / sum(ps)
            else:
                ps = np.zeros(n_categories)
                ps[y[i]] = 1
                Pij[i] = ps
        n = np.sum(Pij, axis=0)

        new_means = np.array([
            np.dot(Pij[:, cat_num], X) / n[cat_num]
            for cat_num in range(n_categories)
        ])
        diff = np.max(np.abs(means - new_means))
        means = new_means

        new_qs = n / float(num_samples)
        diff = max(np.max(np.abs(qs - new_qs)), diff)
        qs = new_qs

        old_covs = covs
        covs = np.zeros((n_categories, n_features, n_features))
        for t in range(num_samples):
            for cat_num in range(n_categories):
                covs[cat_num] += Pij[t, cat_num] * np.outer(X[t] - means[cat_num], X[t] - means[cat_num])

        for cat_num in range(n_categories):
            covs[cat_num] /= n[cat_num]

        diff = max(np.max(np.abs(old_covs - covs)), diff)
        print(f"{iters + 1} / {num_iterations} diff = {diff}")
    return Pij, [means, covs, qs]


probs, [means, covs, qs] = get_probs_ssgmm(cleaned_X_all, y_all, num_iterations=7)


# %% [markdown]
# ### GMM results analysis

# %%
def predict_proba(X, means, covs, qs):
    num_samples, n_features = X.shape
    n_categories = len(unique_labels)
    Pij = np.zeros((num_samples, n_categories))
    for i in range(num_samples):
        ps = np.array([
            multivariate_normal.pdf(X[i], means[cat_num], covs[cat_num], allow_singular=True) * q
            for cat_num, q in zip(range(n_categories), qs)
        ])
        Pij[i] = ps / sum(ps)
        
        if (i + 1) % 500 == 0:
            print(f"Step {i+1}/{num_samples}")
    return Pij

gmm_y_proba = predict_proba(cleaned_X, means, covs, qs)

gmm_y_pred = np.array([label_mapping.inverse[label] for label in np.argmax(gmm_y_proba, axis=-1)])
    
print(classification_report(y, gmm_y_pred))
    
cm = confusion_matrix(y, gmm_y_pred)
sns.heatmap(cm, annot=True)
plt.show()


# %%
gmm_y_all_proba = predict_proba(cleaned_X_all, means, covs, qs)
gmm_y_all_pred = np.array([label_mapping.inverse[label] for label in np.argmax(gmm_y_all_proba, axis=-1)])


print("predicted")
sns.scatterplot(
    x='c1', 
    y='c2',
      hue='category',
    size='correct',
    data=pd.DataFrame({
      'c1': X_pca_all[:, 0],
      'c2': X_pca_all[:, 1],
      'category': list(map(lambda x: categories.get(int(x), "undefined"),
                          gmm_y_all_pred)),
        'correct': list(map(
            lambda x : 15 if x[0] == x[1] else 1, zip(y_all, gmm_y_all_pred)))
  }))

plt.show()

# %%
correct_labels_df = pd.read_csv(os.path.join("..", "data", "labels_trending"))

plot_validation(correct_labels_df[mask_no_missing].category_id_true, gmm_y_all_pred[mask_no_missing.values])

# %%
from scipy import stats

# #############################################################################
# Calculate uncertainty values for each transduced distribution
pred_entropies = stats.distributions.entropy(gmm_y_all_proba.T)
print(pred_entropies.shape)

sns.distplot(pred_entropies)
plt.show()

# %%
transductions_entropies = list(zip(
    gmm_y_all_pred, 
    pred_entropies,
    [i for i in range(len(pred_entropies))]
))

for c in label_spread.classes_:
    print("\nCATEGORY", categories.get(c))
    print(">>> SUPPORT: ", len(list(filter(lambda x : x == c, y_all))), "\n")
    
    t_e_per_class = list(filter(lambda x : x[0] == c, transductions_entropies))
    t_e_per_class = list(sorted(t_e_per_class, key=lambda x : x[1]))
    for _, entropy, idx in t_e_per_class[:5]:
        print(entropy)
        vid_id = agg_df.loc[idx, ["video_id"]].values[0]
        select_from_df = df[df["video_id"] == vid_id]
        if select_from_df.shape[0] > 0:
            print(select_from_df.loc[:, ["title"]].values[0][0])
            print(select_from_df.loc[:, ["tags"]].values[0][0][:100])
            print()

# %% [markdown]
# ## Least certain

# %%
transductions_entropies = list(zip(
    gmm_y_all_pred, 
    pred_entropies,
    [i for i in range(len(pred_entropies))]
))

for c in label_spread.classes_:
    print("\nCATEGORY", categories.get(c))
    print(">>> SUPPORT: ", len(list(filter(lambda x : x == c, y_all))), "\n")
    
    t_e_per_class = list(filter(lambda x : x[0] == c, transductions_entropies))
    t_e_per_class = list(sorted(t_e_per_class, key=lambda x : -1*x[1]))
    for _, entropy, idx in t_e_per_class[:5]:
        print(entropy)
        vid_id = agg_df.loc[idx, ["video_id"]].values[0]
        select_from_df = df[df["video_id"] == vid_id]
        print(select_from_df.loc[:, ["title"]].values[0][0])
        print(select_from_df.loc[:, ["tags"]].values[0][0])
        print()

# %% [markdown]
# ## Most certain

# %%
transductions_entropies = list(zip(
    gmm_y_all_pred, 
    pred_entropies,
    [i for i in range(len(pred_entropies))]
))

for c in label_spread.classes_:
    print("\nCATEGORY", categories.get(c))
    print(">>> SUPPORT: ", len(list(filter(lambda x : x == c, y_all))), "\n")
    
    t_e_per_class = list(filter(lambda x : x[0] == c, transductions_entropies))
    t_e_per_class = list(sorted(t_e_per_class, key=lambda x : x[1]))
    for _, entropy, idx in t_e_per_class[:5]:
        print(entropy)
        vid_id = agg_df.loc[idx, ["video_id"]].values[0]
        select_from_df = df[df["video_id"] == vid_id]
        if select_from_df.shape[0] > 0:
            print(select_from_df.loc[:, ["title"]].values[0][0])
            print(select_from_df.loc[:, ["tags"]].values[0][0][:100])
            print()
