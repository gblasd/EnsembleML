# Boosting
# Train an ensemble of predictors sequentially
# Each predictor tries to correct its predecesor

# Adaboost: Stands for Adaptative Boosting
# Each predictor pay more attention to the instances wrongly predicted by its predecessor.
# Achieved by changing the weights of training instances.
# Each predictor is assigned a coefficient alpha.
# alpha depends on the predictor's training error.

# Classification: Weighted majority voting.
# Regression: Weighted average.

# Import models and utility functions
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Set seed for reproductibility
SEED = 1

# Set the DataFrames
X = pd.DataFrame()
y = pd.DataFrame()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED)

# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)

# Instantiate an Adaboost classifier 'adab_clf'
ada_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)

# Fit 'ada_clf' to the training set
ada_clf.fit(X_train, y_train)

# Predict the test set probabilities of positive class
y_pred_proba = ada_clf.predict_proba(X_test)[:,1]

# Evaluate test-set roc_auc_score
ada_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)

# Print adb_clf_roc_auc_score
print("ROC AUC score: {.2f}".format(ada_clf_roc_auc_score))