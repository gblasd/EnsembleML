import xgboost as xgb
import pandas as pd

churn_data = pd.read_csv("classification_data.csv")

# X, y = churn_data.iloc[:,:-1], churn_data.month_5_still_here
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the DMatrix from X and Y: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"binary_logistic", "max_depth":4}

# Perform cross-valdation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, 
                    params=params, 
                    nfold=4,
                    num_boost_round=10, 
                    metrics="error", 
                    as_pandas=True,
                    seed=123)

# Print cv_results
print(cv_results)

# Print the accuracy
print((1-cv_results["test-error-mean"].iloc[-1]))