import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class_data = pd.read_csv('classification_data.csv')

X, y = class_data.iloc[:, :-1], class_data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123)
xg_cl = xgb.XGBClassifier(objective='binay:logistic',
                          n_estimators=10, 
                          seed=123)

xg_cl.fit(X_train, y_train)

preds = xg_cl.predict(X_test)