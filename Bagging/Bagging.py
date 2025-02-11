# Bagging: Boostrap Aggregation
# Uses a technique know as the boostrap.
# Reduces variance of individual models in the ensemble.
# One algorithm, differents subsets of the training set.

# Import models and utility functions
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Set seed for reproductibility
SEED = 1

# DataFrames
X = pd.DataFrame()
y = pd.DataFrame()

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED
)

# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=4, 
                            min_samples_leaf=0.16, 
                            random_state=SEED)

# Instantiate a BaggingClassifier 'bc'
bc = BaggingClassifier(base_estimator=dt, 
                       n_estimators=300, 
                       obb_score=True, # Out of Bag (OBB) instances
                                       # estimate the performance of the ensemble
                                       # whitout the need for cross-validation
                       n_jobs=-1)

# Fit 'bc' to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate test set accuracy
test_accuracy = accuracy_score(y_test, y_pred)

# Extract the OBB accuracy from 'bc'
obb_accuracy = bc.oob_score_

# Print test set accuracy
print('Accuracy of Bagging Classifier: {.3f}'.format(test_accuracy))

# Print OBB accuracy
print('OBB accuracy: {:.3f}'.format(obb_accuracy))