# Inspection the hyperparameters of a CART in sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

SEED = 1

# Define the DataFrames
X = pd.DataFrame()
y = pd.DataFrame()

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=SEED)

# Set seed to 1 for reproductibility
SEED = 1

# Instantiate a DecisionTreeClasifier 'dt'
dt = DecisionTreeClassifier(random_state=SEED)

# Inspecting the hyperparameters of a CART in sklearn
print(dt.get_params())

# Tuning a CART's Hyperparameters
from sklearn.model_selection import GridSearchCV, roc_auc_score

# Define the grid of hyperparameters 'params_dt'
params_dt = {
    'max_depth': [3, 4.5, 6],
    'min_samples_leaf': [0.04, 0.06, 0.08],
    'max_features': [0.2, 0.4, 0.6, 0.8]
}

# Instantiate a 10-fold CV grid search object 'grid_dt'
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring='accuracy',
                       cv=10,
                       n_jobs=-1)

# Fit 'grid_dt' to the training data
grid_dt.fit(X_train, y_train)

# Extracting the best hyperparameters from 'grid_dt'
best_hyperparams = grid_dt.best_params_
print('Best hyperparameters:\n', best_hyperparams)

# Extract best CV score from 'grid_dt'
best_CV_score = grid_dt.best_score_
print('Best CV accuracy: {}'.format(best_CV_score))

# Extract best model from 'grid_dt'
best_model = grid_dt.best_estimator_

# Predict the set probabilities of the positive class
y_pred_proba = grid_dt.predict_proba(X_test)[:,1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print("Test set ROC AUC score: {:.3f}".format(test_roc_auc))

# Evaluate test set accuracy
test_acc = best_model.score(X_test, y_test)

# Print test set accuracy
print("Test accuracy: {}".format(test_acc))