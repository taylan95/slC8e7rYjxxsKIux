#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
import itertools
import scipy.stats as stats

from scipy.stats import chi2_contingency
from feature_engine.outlier_removers import Winsorizer
from feature_engine.discretisers import EqualWidthDiscretiser
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score, mean_squared_error, roc_curve, confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

csv_file = "term-deposit-marketing-2020.csv"


def get_data_from_csv(file_name):
    data = pd.read_csv(file_name)
    return data.copy()


df = get_data_from_csv(csv_file)


def print_summary_data(data):
    print("Rows         :", data.shape[0])
    print("-" * 30)
    print("Columns      :", data.shape[1])
    print("-" * 30)
    print("Features :\n ", data.columns.tolist())
    print("-" * 30)
    print("Missing Values :   ", data.isnull().sum().values.sum())
    print("-" * 30)
    print("Unique Values:     \n", data.nunique())


print_summary_data(df)


def get_descriptive_info(data):
    print("Info      :\n", data.info())
    print("-" * 30)
    print("Describe    :\n", data.describe())
    print("-" * 30)
    print("Uniques     :\n")
    for i in data.columns:
        if data[i].dtype == "O":
            print(data[i].unique())


get_descriptive_info(df)


def draw_target_value_pie_chart(data, label_1, label_2, variable, title):
    labels = label_1, label_2
    sizes = [data[variable][data[variable] == label_1].count(), data[variable][data[variable] == label_2].count()]
    explode = (0, 0.1)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%",
            shadow=True, startangle=90)
    ax1.axis("equal")
    plt.title(title, size=20)
    plt.show()


draw_target_value_pie_chart("no", "yes", "y", "Proportion of customer buying deposit or not buying")

sns.set(style="ticks", color_codes=True)

fig, axes = plt.subplots(nrows=2, ncols=ncols_value, figsize=(25, 15))
sns.countplot(x="marital", data=df, hue="y", ax=axes[0][0])
sns.countplot(x="default", data=df, hue="y", ax=axes[0][1])
sns.countplot(x="housing", data=df, hue="y", ax=axes[0][2])
sns.countplot(x="loan", data=df, hue="y", ax=axes[1][0])
sns.countplot(x="contact", data=df, hue="y", ax=axes[1][1])
sns.countplot(x="month", data=df, hue="y", ax=axes[1][2])
plt.show(fig)

sns.pairplot(data=df, hue="y")


def get_chi_square(data, target, variable_list):  #  y , ["education", "job"]
    for var in variable_list:
        csq = chi2_contingency(pd.crosstab(data[target], data[var]))
        print("P-value: ", csq[1])


get_chi_square("y", ["education", "job"])

fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="y", data=df, hue="education")
plt.title("Impact of Education on Target")
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="y", data=df, hue="job")
plt.title("Impact of Job on Target")
plt.show()


def group_age_and_draw_chart(group_name, target):
    bins = np.arange(10, 100, 10)

    df[group_name] = np.digitize(df.age, bins, right=True)

    counts = df.groupby([group_name, target]).age.count().unstack()
    print(counts)

    ax = counts.plot(kind="bar", stacked=False, colormap="Paired")

    for p in ax.patches:
        ax.annotate(
            np.round(p.get_height(), decimals=0).astype(np.int64),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha="center",
            va="center",
            xytext=(2, 10),
            textcoords="offset points"
        )

    plt.xlabel("Age Group")
    plt.ylabel("Co-Occurences ")
    plt.title("Comparison Of Occurences  In An Age Group", fontsize=14)
    plt.show()


group_age_and_draw_chart("type", "y")

fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.boxplot(y="balance", x="y", hue="y", data=df, ax=axarr[0][0])
sns.boxplot(y="age", x="y", hue="y", data=df, ax=axarr[0][1])
sns.boxplot(y="duration", x="y", hue="y", data=df, ax=axarr[1][0])
sns.boxplot(y="day", x="y", hue="y", data=df, ax=axarr[1][1])

g = sns.FacetGrid(df, col="y")
g.map(sns.distplot, "age", bins=25)
plt.show()

g = sns.FacetGrid(df, col="y")
g.map(sns.distplot, "balance", bins=25)
plt.show()

df[["age"]].hist(bins=30, figsize=(8, 4))
plt.show()

df[["duration"]].hist(bins=30, figsize=(8, 4))
plt.show()

df["y"] = np.where(df["y"] == "yes", 1, 0)

df["education"] = np.where(df["education"] == "unknown", df["education"].mode(), df["education"])
df["job"] = np.where(df["job"] == "unknown", "missing", df["job"])
df["contact"] = np.where(df["contact"] == "unknown", "other", df["contact"])


def deeply_plots(df, variable_list):
    for variable in variable_list:
        print(variable)
        plt.figure(figsize=(16, 4))
        # histogram
        plt.subplot(1, 3, 1)
        sns.distplot(df[variable], bins=30)
        plt.title("Histogram")
        # QQ-plot
        plt.subplot(1, 3, 2)
        stats.probplot(df[variable], dist="norm", plot=plt)
        plt.ylabel("quantiles")
        # boxplot
        plt.subplot(1, 3, 3)
        sns.boxplot(y=df[variable])
        plt.title("Boxplot")
        plt.show()


deeply_plots(df, ["balance", "duration"])

wind = Winsorizer(distribution="skewed",
                  tail="both",
                  fold=1.5,
                  variables=["balance", "duration"])

wind.fit(df)
df = wind.transform(df)

deeply_plots(df, "balance"), deeply_plots(df, "duration")

dummylist = []

dummy_variables = ["job", "marital", "education", "default", "housing", "loan", "contact", "month"]
for var in dummy_variables:
    dummylist.append(pd.get_dummies(df[var], prefix=var, prefix_sep="_", drop_first=True))
    dummies_collected = pd.concat(dummylist, axis=1)

df.drop(dummy_variables, axis=1, inplace=True)
df = pd.concat([df, dummies_collected], axis=1)

df.drop("type", axis=1, inplace=True)
df.shape

X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

countlabel = ["age", "day", "campaign"]
disc = EqualWidthDiscretiser(bins=10, variables=countlabel)

disc.fit(X_train)
disc.fit(X_test)

disc.binner_dict_

X_train = disc.transform(X_train)
X_test = disc.transform(X_test)


def bingraph(train, test, variable):
    t1 = train.groupby([variable])[variable].count() / len(train)
    t2 = test.groupby([variable])[variable].count() / len(test)

    tmp = pd.concat([t1, t2], axis=1)
    tmp.columns = ["train", "test"]
    tmp.plot.bar()
    plt.xticks(rotation=0)
    plt.ylabel("number of observations per bin")


bingraph(X_train, X_test, "age"),
bingraph(X_train, X_test, "day"),
bingraph(X_train, X_test, "campaign")

dummylist_train = []
dummylist_test = []
countlabel2 = ["age", "day", "campaign"]

for lab in countlabel2:
    dummylist_train.append(pd.get_dummies(X_train[lab], prefix=lab, prefix_sep="_", drop_first=True))
    dummies_collected_train = pd.concat(dummylist_train, axis=1)

X_train.drop(countlabel2, axis=1, inplace=True)
X_train = pd.concat([X_train, dummies_collected_train], axis=1)

for lab in countlabel2:
    dummylist_test.append(pd.get_dummies(X_test[lab], prefix=lab, prefix_sep="_", drop_first=True))
    dummies_collected_test = pd.concat(dummylist_test, axis=1)

X_test.drop(countlabel2, axis=1, inplace=True)
X_test = pd.concat([X_test, dummies_collected_test], axis=1)

X_train.shape, X_test.shape

class FeatureSelector:

    def __init__(self, X_train):
        self.X_train = X_train

    def get_correlation_matrix(self):
        corr_matrix = self.X_train.corr()
        fig, ax = plt.subplots(figsize=(20, 15))
        ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

    @staticmethod
    def correlation(dataset, threshold):
        col_corr = set()
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]
                    col_corr.add(colname)
                    
        return col_corr

    def get_corr_features_len(self):
        corr_features = self.correlation(self.X_train, 0.8)
        return len(set(corr_features))

    def get_constant_features_len(self):
        constant_features = [
                feat for feat in self.X_train.columns if self.X_train[feat].std() == 0
            ]
        return len(constant_features)

    def get_duplicated_feat_len(self):
        duplicated_feat = []
        for i in range(0, len(self.X_train.columns)):
            if i % 10 == 0:
                print(i)
            col_1 = self.X_train.columns[i]
            for col_2 in self.X_train.columns[i + 1:]:
                if self.X_train[col_1].equals(self.X_train[col_2]):
                    duplicated_feat.append(col_2)

        return len(set(duplicated_feat))

    def get_roc_values(self):
        roc_values = []
        for feature in self.X_train.columns:
            clf = RandomForestClassifier()
            clf.fit(self.X_train[feature].fillna(0).to_frame(), y_train)
            y_scored = clf.predict_proba(X_test[feature].fillna(0).to_frame())
            roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))

        roc_values = pd.Series(roc_values)
        roc_values.index = self.X_train.columns
        roc_values.sort_values(ascending=False)

        roc_values.sort_values(ascending=False).plot.bar(figsize=(20, 8))
        return roc_values

feature_selector = FeatureSelector(X_train)

feature_selector.get_correlation_matrix()

feature_selector.get_corr_features_len()
feature_selector.get_constant_features_len()
feature_selector.get_duplicated_feat_len()

roc_values = feature_selector.get_roc_values()

len(roc_values[roc_values > 0.5])


drop_list = roc_values[roc_values < 0.5].index

X_train.drop(drop_list, axis=1, inplace=True)
X_test.drop(drop_list, axis=1, inplace=True)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# logistic
logistic = LogisticRegression(random_state=0)
logistic.fit(X_train, y_train)

# knn
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
knn.fit(X_train, y_train)

# svm
svm = SVC(kernel="linear", random_state=0, probability=True)
svm.fit(X_train, y_train)

# dtree
dtree = DecisionTreeClassifier(criterion="entropy", random_state=0)
dtree.fit(X_train, y_train)

# rf
rf = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
rf.fit(X_train, y_train)

# xgboost
xgc = XGBClassifier()
xgc.fit(X_train, y_train)


modellist = [logistic, knn, svm, dtree, rf, xgc]
for model in modellist:
    print(model.score(X_train, y_train))


modellist = [logistic, knn, svm, dtree, rf, xgc]
length = len(modellist)
mods = ["Logistic Regression", "KNN Classifier", "SVM Classifier Linear",
        "Decision Tree", "Random Forest Classifier", "XGBoost Classifier"]

fig = plt.figure(figsize=(13, 15))
plt.rcParams["figure.facecolor"] = "white"
for i, j, k in itertools.zip_longest(modellist, range(length), mods):
    plt.subplot(3, 3, j + 1)
    predictions = i.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    sns.heatmap(conf_matrix, annot=True, fmt="d", square=True,
                xticklabels=["no", "yes"],
                yticklabels=["no", "yes"],
                linewidths=2, linecolor="w", cmap=plt.cm.Blues)
    plt.ylim(-0.05, 2)
    plt.title(k, color="b")
    plt.subplots_adjust(wspace=.3, hspace=.3)


for i, k in zip(modellist, mods):
    predictions = i.predict(X_test)
    print("{}".format(k))
    print(classification_report(y_test, predictions))

clf = RandomForestClassifier()

param_grid = {
    "n_estimators": [100, 300, 500, 800, 1000],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [4, 5, 6, 7, 8, 10, 15, 20],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 5, 10, 15, 100],
    "min_samples_leaf": [1, 2, 5, 10]
}

CV_Random_clf = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, cv=5,
                                   verbose=2, random_state=42, n_jobs=-1)


CV_Random_clf.fit(X_train, y_train)

CV_Random_clf.best_params_

rfc = RandomForestClassifier(random_state=42, max_features="log2", n_estimators=300, max_depth=15, criterion="entropy",
                             min_samples_split=2, min_samples_leaf=2)


rfc.fit(X_train, y_train)


rfc.score(X_train, y_train)

y_pred = rfc.predict(X_test)

print("Accuracy for Random Forest on CV data: ", accuracy_score(y_test, y_pred))

y_pred_proba = rfc.predict_proba(X_test)
y_pred_proba_y = y_pred_proba[:, 1]
AUROC = roc_auc_score(y_test, y_pred_proba_y)

label = "AUROC = " + str(np.around(roc_auc_score(y_test, y_pred_proba_y), 3))
print(label)
plt.style.use("classic")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_y)
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle="--", color="k")
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC_Curve")

df.head()

df_clus = df.copy()

most_important = roc_values.sort_values(ascending=False)

most_important.index[0:7]

df_clus = df_clus[most_important.index[0:7]]

df_clus["y"] = df["y"]

model = (KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300,
                tol=0.0001, random_state=111, algorithm="elkan"))
model.fit(df_clus)
labels2 = model.labels_

plt.figure(1, figsize=(15, 6))
plt.plot(np.arange(1, 11), inertia, "o")
plt.plot(np.arange(1, 11), inertia, "-", alpha=0.5)
plt.xlabel("Number of Clusters"), plt.ylabel("Inertia")
plt.show()

inertia = []
for n in range(1, 11):
    model = (KMeans(n_clusters=n, init="k-means++", n_init=10, max_iter=300,
                    tol=0.0001, random_state=111, algorithm="elkan"))
    model.fit(df_clus)
    inertia.append(model.inertia_)

labels2

df_clus["label"] = labels2

df_clus.head()


function = {"y": "sum", "y": "count"}

investment_selling = df_clus.groupby(["label", "y"]).size().reset_index(name="count")

investment_selling

y_0 = investment_selling.loc[investment_selling["y"] == 0, "count"].reset_index().drop("index", axis=1)


y_1 = investment_selling.loc[investment_selling["y"] == 1, "count"].reset_index().drop("index", axis=1)

y_labels = pd.concat([y_0, y_1], axis=1)

y_labels.columns = ["count_y0", "count_y1"]

y_labels

y_labels["most_selling"] = y_labels["count_y1"] / (y_labels["count_y0"] + y_labels["count_y1"])

y_labels["segment"] = ["segment_1", "segment_2", "segment_3", "segment_4"]

y_labels = y_labels[["segment", "count_y0", "count_y1", "most_selling"]]

y_labels

y_labels["segment"] = ["middlesegment", "bestsegment", "worsesegment", "middlesegment"]
y_labels
