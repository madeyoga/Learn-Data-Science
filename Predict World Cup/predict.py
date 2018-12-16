import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

final = pd.read_csv('final.csv', index_col=0)

X = final.drop(['winning_team'], axis=1)
y = final['winning_team']
y = y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.33,
    random_state = 8
    )

print(final.head())

## Logistic Regression
print("=====================================\nLogistic Regression:")
log_reg = LogisticRegression(
    solver='lbfgs',
    multi_class='auto',
    max_iter=1500 ## lbfgs failed to converge. Increase the number of iterations.
    )
log_reg.fit(X_train, y_train)

##training_score = log_reg.score(X_train, y_train)
##test_score = log_reg.score(X_test, y_test)

training_score = accuracy_score(
    log_reg.predict(X_train),
    y_train
    )

test_score = accuracy_score(
    log_reg.predict(X_test),
    y_test
    )

print("Training Accuracy: %.2f%%" % (training_score * 100))
print("Test Accuracy: %.2f%%" % (test_score * 100))
print("CrossValidation Accuracy: %.2f%%" % (
    cross_val_score(log_reg, X_train, y_train, cv=3).mean() * 100
    ))

## Multi-Layer Perceptron
print("=====================================\nMulti-Layer Perceptron:")
mlp = MLPClassifier(
    activation='tanh',
    hidden_layer_sizes=(6,10),
    alpha=1e-3,
    max_iter=3000
    )

mlp.fit(X_train, y_train)

training_score = accuracy_score(
    mlp.predict(X_train),
    y_train
    )

test_score = accuracy_score(
    mlp.predict(X_test),
    y_test
    )

print("Training Accuracy: %.2f%%" % (training_score * 100))
print("Test Accuracy: %.2f%%" % (test_score * 100))
print("CrossValidation Accuracy: %.2f%%" % (
    cross_val_score(mlp, X_train, y_train, cv=3).mean() * 100
    ))

## K-NearestNeighbors
print("=====================================\nK-NearestNeighbors-KDTree:")
knn = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='kd_tree'
    )
knn.fit(X_train, y_train)
training_score = accuracy_score(
    knn.predict(X_train),
    y_train
    )
test_score = accuracy_score(
    knn.predict(X_test),
    y_test
    )

print("Training Accuracy: %.2f%%" % (training_score * 100))
print("Test Accuracy: %.2f%%" % (test_score * 100))
print("CrossValidation Accuracy: %.2f%%" % (
    cross_val_score(knn, X_train, y_train, cv=3).mean() * 100
    ))
print("=====================================")
