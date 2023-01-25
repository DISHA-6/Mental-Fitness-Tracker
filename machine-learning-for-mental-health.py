""" Predictors of mental health illness  """

""" 
The proccess is the following:
1. [Library and data loading](  # Library_and_data_loading)
2. [Data cleaning](  # Data_cleaning)
3. [Encoding data](  # Encoding_data)
4. [Covariance Matrix. Variability comparison between categories of variables](  # Covariance_Matrix)
5. [Some charts to see data relationship](  # Some_charts_to_see_data_relationship)
6. [Scaling and fitting](  # Scaling_and_fitting)
7. [Tuning](  # Tuning)
8. [Evaluating models](  # Evaluating_models)
    1. [Logistic Eegression](  # Logistic_regressio)
    2. [KNeighbors Classifier](  # KNeighborsClassifier)
    3. [Decision Tree Classifier](  # Decision_Tree_classifier)
    4. [Random Forests](  # Random_Forests)
    5. [Bagging](  # Bagging)
    6. [Boosting](  # Boosting)
    7. [Stacking](  # Stacking)
9. [Predicting with Neural Network](  # Predicting_with_Neural_Network)
10. [Success method plot](  # Success_method_plot)
11. [Creating predictions on test set](  # Creating_predictions_on_test_set)
12. [Submission](  # Submission)
13. [Conclusions](  # Conclusions) """


""" 1. Library and data loading """


import tensorflow as tf
import argparse
from subprocess import check_output
from mlxtend.classifier import StackingClassifier
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# reading in CSV's from a file path
train_df = pd.read_csv('../input/survey.csv')


# Pandas: whats the data row count?
print(train_df.shape)

# Pandas: whats the distribution of the data?
print(train_df.describe())

# Pandas: What types of data do i have?
print(train_df.info())

""" 2. Data cleaning """

# missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()
           ).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
print(missing_data)


# dealing with missing data
# Let’s get rid of the variables "Timestamp",“comments”, “state” just to make our lives easier.
train_df = train_df.drop(['comments'], axis=1)
train_df = train_df.drop(['state'], axis=1)
train_df = train_df.drop(['Timestamp'], axis=1)

# just checking that there's no missing data missing...
train_df.isnull().sum().max()
train_df.head(5)


""" Cleaning NaN """

# Assign default values for each data type
defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0

# Create lists by data tpe
intFeatures = ['Age']
stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                  'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                  'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                  'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                  'seek_help']
floatFeatures = []

# Clean the NaN's
for feature in train_df:
    if feature in intFeatures:
        train_df[feature] = train_df[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        train_df[feature] = train_df[feature].fillna(defaultString)
    elif feature in floatFeatures:
        train_df[feature] = train_df[feature].fillna(defaultFloat)
    else:
        print('Error: Feature %s not recognized.' % feature)
train_df.head(5)


# clean 'Gender'
# Slower case all columm's elements
gender = train_df['Gender'].str.lower()
# print(gender)

# Select unique elements
gender = train_df['Gender'].unique()

# Made gender groups
male_str = ["male", "m", "male-ish", "maile", "mal",
            "male (cis)", "make", "male ", "man", "msle", "mail", "malr", "cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender",
             "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
female_str = ["cis female", "f", "female", "woman",  "femake",
              "female ", "cis-female/femme", "female (cis)", "femail"]

for (row, col) in train_df.iterrows():

    if str.lower(col.Gender) in male_str:
        train_df['Gender'].replace(
            to_replace=col.Gender, value='male', inplace=True)

    if str.lower(col.Gender) in female_str:
        train_df['Gender'].replace(
            to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in trans_str:
        train_df['Gender'].replace(
            to_replace=col.Gender, value='trans', inplace=True)

# Get rid of bullshit
stk_list = ['A little about you', 'p']
train_df = train_df[~train_df['Gender'].isin(stk_list)]

print(train_df['Gender'].unique())


# complete missing age with mean
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Fill with media() values < 18 and > 120
s = pd.Series(train_df['Age'])
s[s < 18] = train_df['Age'].median()
train_df['Age'] = s
s = pd.Series(train_df['Age'])
s[s > 120] = train_df['Age'].median()
train_df['Age'] = s

# Ranges of Age
train_df['age_range'] = pd.cut(train_df['Age'], [0, 20, 30, 65, 100], labels=[
                               "0-20", "21-30", "31-65", "66-100"], include_lowest=True)

# There are only 0.014% of self employed so let's change NaN to NOT self_employed
# Replace "NaN" string from defaultString
train_df['self_employed'] = train_df['self_employed'].replace(
    [defaultString], 'No')
print(train_df['self_employed'].unique())


# There are only 0.20% of self work_interfere so let's change NaN to "Don't know
# Replace "NaN" string from defaultString

train_df['work_interfere'] = train_df['work_interfere'].replace(
    [defaultString], 'Don\'t know')
print(train_df['work_interfere'].unique())


""" 3. Encoding data """


# Encoding data
labelDict = {}
for feature in train_df:
    le = preprocessing.LabelEncoder()
    le.fit(train_df[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    train_df[feature] = le.transform(train_df[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] = labelValue

for key, value in labelDict.items():
    print(key, value)

# Get rid of 'Country'
train_df = train_df.drop(['Country'], axis=1)
train_df.head()


""" Testing there aren't any missing data """
# missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()
           ).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
print(missing_data)

""" 4. Covariance Matrix. Variability comparison between categories of variables """
# correlation matrix
corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# treatment correlation matrix
k = 10  # number of variables for heatmap
cols = corrmat.nlargest(k, 'treatment')['treatment'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={
                 'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


""" 5. Some charts to see data relationship """


# Distribiution and density by Age
plt.figure(figsize=(12, 8))
sns.distplot(train_df["Age"], bins=24)
plt.title("Distribuition and density by Age")
plt.xlabel("Age")


# Separate by treatment or not

g = sns.FacetGrid(train_df, col='treatment', size=5)
g = g.map(sns.distplot, "Age")


# Let see how many people has been treated
plt.figure(figsize=(12, 8))
labels = labelDict['label_Gender']
g = sns.countplot(x="treatment", data=train_df)
g.set_xticklabels(labels)

plt.title('Total Distribuition by treated or not')


""" Draw a nested barplot to show probabilities for class and sex """


o = labelDict['label_age_range']

g = sns.factorplot(x="age_range", y="treatment", hue="Gender", data=train_df,
                   kind="bar",  ci=None, size=5, aspect=2, legend_out=True)
g.set_xticklabels(o)

plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Age')
# replace legend labels

new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# Positioning the legend
g.fig.subplots_adjust(top=0.9, right=0.8)

plt.show()


""" Barplot to show probabilities for family history """


o = labelDict['label_family_history']
g = sns.factorplot(x="family_history", y="treatment", hue="Gender",
                   data=train_df, kind="bar", ci=None, size=5, aspect=2, legend_out=True)
g.set_xticklabels(o)
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Family History')

# replace legend labels
new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# Positioning the legend
g.fig.subplots_adjust(top=0.9, right=0.8)

plt.show()


""" Barplot to show probabilities for care options """

o = labelDict['label_care_options']
g = sns.factorplot(x="care_options", y="treatment", hue="Gender",
                   data=train_df, kind="bar", ci=None, size=5, aspect=2, legend_out=True)
g.set_xticklabels(o)
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Care options')

# replace legend labels
new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# Positioning the legend
g.fig.subplots_adjust(top=0.9, right=0.8)
plt.show()


""" Barplot to show probabilities for benefits """

o = labelDict['label_benefits']
g = sns.factorplot(x="care_options", y="treatment", hue="Gender",
                   data=train_df, kind="bar", ci=None, size=5, aspect=2, legend_out=True)
g.set_xticklabels(o)
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Benefits')

# replace legend labels
new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# Positioning the legend
g.fig.subplots_adjust(top=0.9, right=0.8)
plt.show()


""" Barplot to show probabilities for work interfere """

o = labelDict['label_work_interfere']
g = sns.factorplot(x="work_interfere", y="treatment", hue="Gender",
                   data=train_df, kind="bar", ci=None, size=5, aspect=2, legend_out=True)
g.set_xticklabels(o)
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Work interfere')

# replace legend labels
new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# Positioning the legend
g.fig.subplots_adjust(top=0.9, right=0.8)
plt.show()


""" 6. Scaling and fitting """


# Scaling Age
scaler = MinMaxScaler()
train_df['Age'] = scaler.fit_transform(train_df[['Age']])
train_df.head()


# define X and y
feature_cols = ['Age', 'Gender', 'family_history', 'benefits',
                'care_options', 'anonymity', 'leave', 'work_interfere']
X = train_df[feature_cols]
y = train_df.treatment

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0)

# Create dictionaries for final graph
# Use: methodDict['Stacking'] = accuracy_score
methodDict = {}
rmseDict = ()


# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

labels = []
for f in range(X.shape[1]):
    labels.append(feature_cols[f])

# Plot the feature importances of the forest
plt.figure(figsize=(12, 8))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), labels, rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.show()

""" 
7. Tuning 
Evaluating a Classification Model.
This function will evalue
Classification accuracy
percentage of correct predictions """


""" 8. Evaluating models """

# Logistic Regression


def logisticRegression():
    # train a logistic regression model on the training set
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = logreg.predict(X_test)

    print('########### Logistic Regression ###############')

    accuracy_score = evalClassModel(logreg, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['Log. Regres.'] = accuracy_score * 100


logisticRegression()


# KNeighbors Classifier

def Knn():
    # Calculating the best parameters
    knn = KNeighborsClassifier(n_neighbors=5)

    # define the parameter values that should be searched
    k_range = list(range(1, 31))
    weight_options = ['uniform', 'distance']

    # specify "parameter distributions" rather than a "parameter grid"
    param_dist = dict(n_neighbors=k_range, weights=weight_options)
    tuningRandomizedSearchCV(knn, param_dist)

    # train a KNeighborsClassifier model on the training set
    knn = KNeighborsClassifier(n_neighbors=27, weights='uniform')
    knn.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = knn.predict(X_test)

    print('########### KNeighborsClassifier ###############')

    accuracy_score = evalClassModel(knn, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['KNN'] = accuracy_score * 100


""" KNEIGHBORSCLASSIFIER """

Knn()


# Decision Tree classifier

def treeClassifier():
    # Calculating the best parameters
    tree = DecisionTreeClassifier()
    featuresSize = feature_cols.__len__()
    param_dist = {"max_depth": [3, None],
                  "max_features": randint(1, featuresSize),
                  "min_samples_split": randint(2, 9),
                  "min_samples_leaf": randint(1, 9),
                  "criterion": ["gini", "entropy"]}
    tuningRandomizedSearchCV(tree, param_dist)

    # train a decision tree model on the training set
    tree = DecisionTreeClassifier(
        max_depth=3, min_samples_split=8, max_features=6, criterion='entropy', min_samples_leaf=7)
    tree.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = tree.predict(X_test)

    print('########### Tree classifier ###############')

    accuracy_score = evalClassModel(tree, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['Tree clas.'] = accuracy_score * 100

    treeClassifier()

    # Random Forests

    def randomForest():
        # Calculating the best parameters
    forest = RandomForestClassifier(n_estimators=20)

    featuresSize = feature_cols.__len__()
    param_dist = {"max_depth": [3, None],
                  "max_features": randint(1, featuresSize),
                  "min_samples_split": randint(2, 9),
                  "min_samples_leaf": randint(1, 9),
                  "criterion": ["gini", "entropy"]}
    tuningRandomizedSearchCV(forest, param_dist)

    # Building and fitting my_forest
    forest = RandomForestClassifier(
        max_depth=None, min_samples_leaf=8, min_samples_split=2, n_estimators=20, random_state=1)
    my_forest = forest.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = my_forest.predict(X_test)

    print('########### Random Forests ###############')

    accuracy_score = evalClassModel(my_forest, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['R. Forest'] = accuracy_score * 100

    randomForest()

    # Bagging

    def bagging():
        # Building and fitting
    bag = BaggingClassifier(DecisionTreeClassifier(
    ), max_samples=1.0, max_features=1.0, bootstrap_features=False)
    bag.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = bag.predict(X_test)

    print('########### Bagging ###############')

    accuracy_score = evalClassModel(bag, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['Bagging'] = accuracy_score * 100

    bagging()

    # Boosting

    def boosting():
        # Building and fitting
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
    boost = AdaBoostClassifier(base_estimator=clf, n_estimators=500)
    boost.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = boost.predict(X_test)

    print('########### Boosting ###############')

    accuracy_score = evalClassModel(boost, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['Boosting'] = accuracy_score * 100

    boosting()


# Stacking

def stacking():
    # Building and fitting
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()
    stack = StackingClassifier(
        classifiers=[clf1, clf2, clf3], meta_classifier=lr)
    stack.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = stack.predict(X_test)

    print('########### Stacking ###############')

    accuracy_score = evalClassModel(stack, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['Stacking'] = accuracy_score * 100

    stacking()

# **9. Predicting with Neural Network**

# Create input functions


batch_size = 100
train_steps = 1000

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

# Define the feature columns


# Define Tensorflow feature columns
age = tf.feature_column.numeric_column("Age")
gender = tf.feature_column.numeric_column("Gender")
family_history = tf.feature_column.numeric_column("family_history")
benefits = tf.feature_column.numeric_column("benefits")
care_options = tf.feature_column.numeric_column("care_options")
anonymity = tf.feature_column.numeric_column("anonymity")
leave = tf.feature_column.numeric_column("leave")
work_interfere = tf.feature_column.numeric_column("work_interfere")
feature_columns = [age, gender, family_history, benefits,
                   care_options, anonymity, leave, work_interfer]

# Instantiate an Estimator

# Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.
model = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                   hidden_units=[10, 10],
                                   optimizer=tf.train.ProximalAdagradOptimizer(
                                       learning_rate=0.1,
                                       l1_regularization_strength=0.001
                                   ))


# **10. Success method plot**


def plotSuccess():
    s = pd.Series(methodDict)
    s = s.sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    # Colors
    ax = s.plot(kind='bar')
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(), 2)),
                    (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.ylim([70.0, 90.0])
    plt.xlabel('Method')
    plt.ylabel('Percentage')
    plt.title('Success of methods')

    plt.show()

    plotSuccess()

    # **11. Creating predictions on test set**

    # Generate predictions with the best method
clf = AdaBoostClassifier()
clf.fit(X, y)
dfTestPredictions = clf.predict(X_test)

# Write predictions to csv file
# We don't have any significative field so we save the index
results = pd.DataFrame({'Index': X_test.index, 'Treatment': dfTestPredictions})
# Save to file
# This file will be visible after publishing in the output section
results.to_csv('results.csv', index=False)
results.head()


""" 
                            END MACHINE LEARNING MENTAL HEALTH PREDICTION PROJECT  """
