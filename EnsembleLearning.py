
# Ensemble Learning
# Train different models in the same dataset
# Let each model make its predictions
# Meta-model: aggregates predictins of individual models.
# Final predition: more robust and less prone to errors.
# Best results: models


# Import functins to compute accuracy and split data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Import models, including VotingClassifier meta-model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier

# Set seed for repoductibility
SEED = 1

# Split data into 70% traon and 30% test
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Instantiate individual classifiers
lr = LogisticRegression(random_state=SEED)
knn = KNN() # n_neightbors=27
dt = DecisionTreeClassifier(random_state=SEED) # min_samples_leaf=0.13

# Define a list called classifier that contains the tuples (classifier_name, classifier)
classifiers = [('Logistic Regression', lr),
               ('K Nearest Neighbours', knn),
               ('Classificaciont Tree', dt)]

# Interate over the defined list of tuples containing the classifiers
for cls_name, clf in classifiers:
    # fit cls tot he training set
    clf.fit(X_train, y_train)

    # Predict the labels of the test set
    y_pred = clf.predict(X_test)

    # Evaluate the accuracy of clf on the test set
    print('{s:} : {:.3f}'.format(cls_name, accuracy_score(y_test, y_pred)))

# Instantiate a VotingClassifier 'vc'
vc = VotingClassifier(estimators=classifiers)

# Fit 'vc' to the training set and predict test set labels
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)

# Evaluate the test-set accuracy of 'vc'
print('Voting Classifier: {.3}'.format(accuracy_score(y_test, y_pred)))