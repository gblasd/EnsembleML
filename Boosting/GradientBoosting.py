# Gradient Boosted Trees

# Sequential correction of predecessor's errors.
# Does not tweak the weights of training instances.
# Fit each predictor is trained using its predecessor's residual error as labels
# Gradient Boosted Tress: a CART is used as a base learner.

# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
import pandas as pd

# Set seed for reproductibility
SEED = 1

# Set the DataFrames
X = pd.DataFrame()
y = pd.DataFrame()

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED)

# instantiate a GradientBoostingRegressor 'gbt'
gbt = GradientBoostingRegressor(n_estimators=300,
                                max_depth=1,
                                random_state=SEED)

# Fit 0gbt' to the training set
gbt.fit(X_train, y_train)

# Predict the test set labels
y_pred = gbt.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)

# Print the test set RMSE
print('Test set RMSE: {:.2f}'.format(rmse_test))

# Gradient Boosting: Cons

# GB involves an exhaustive search procedure.
# Each CART is trained to find the best split points and features.