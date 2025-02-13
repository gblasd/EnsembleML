import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, mean_squared_error

boston_data = pd.read_csv("boston_housing.csv")
X, y = boston_data.iloc[:,:-1], boston_data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123)

xg_reg = xgb.XGBRegressor(objective='reg:aquarederror', 
                          n_estimator=10, 
                          seed=10)

xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))


############################################################

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

boston_data = pd.read_csv("boston_housing.csv")

X, y = boston_data.iloc[:,:-1], boston_data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123)

DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test  = xgb.DMatric(data=X_test, label=y_test)

params = {"booster":"gblinear", "objective":"reg:squarederror"}

xg_reg = xgb.train(params = params,
                   dtrain = DM_train,
                   num_boost_round = 10)

preds = xg_reg.predict(DM_test)