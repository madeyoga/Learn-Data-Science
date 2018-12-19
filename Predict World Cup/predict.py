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
    random_state=8
    )

print(final.head())

## Logistic Regression
print("=====================================\nLogistic Regression:")
log_reg = LogisticRegression(
    solver='lbfgs',
    multi_class='auto',
    max_iter=3000 ## lbfgs failed to converge. Increase the number of iterations.
    )
log_reg.fit(X_train, y_train)

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
    hidden_layer_sizes=(5,2),
    alpha=1e-3,
    max_iter=3000,
    random_state=8
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
    n_neighbors=1,
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

ranking = pd.read_csv('Datasets/fifa_rankings.csv')
fixtures = pd.read_csv('Datasets/fixtures.csv')

pred_set = []
### first post -> home_team rank
### second post -> away_team rank
fixtures.insert(
    1, 'first_position',
    fixtures['Home Team'].map(ranking.set_index('Team')['Position'])
    )
fixtures.insert(
    2, 'second_position',
    fixtures['Away Team'].map(ranking.set_index('Team')['Position'])
    )
fixtures = fixtures.iloc[:48, :]
## print(fixtures.head())
### home_team -> favorite, has higher FIFA rank
for index, row in fixtures.iterrows():
    if row['first_position'] < row['second_position']:        
        if row['Result'] == row['Home Team']:
            win_id = 2
        elif row['Result'] == row['Away Team']:
            win_id = 0
        else:
            win_id = 1
        pred_set.append({'home_team': row['Home Team'], 'away_team': row['Away Team'], 'winning_team': win_id})
    else:
        if row['Result'] == row['Home Team']:
            win_id = 0
        elif row['Result'] == row['Away Team']:
            win_id = 2
        else:
            win_id = 1
        pred_set.append({'home_team': row['Away Team'], 'away_team': row['Home Team'], 'winning_team': win_id})
        
pred_set = pd.DataFrame(pred_set)
backup_pred_set = pred_set
y_pred_set = pred_set['winning_team']
## print(pred_set.head())


# Get dummy variables and drop winning_team column
pred_set = pd.get_dummies(pred_set, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

# Add missing columns compared to the model's training dataset
missing_cols = set(final.columns) - set(pred_set.columns)
for c in missing_cols:
    pred_set[c] = 0
pred_set = pred_set[final.columns]

# Remove winning team column
pred_set = pred_set.drop(['winning_team'], axis=1)

## print(pred_set.head())

###group matches 
predictions = log_reg.predict(pred_set)
for i in range(fixtures.shape[0]):
    print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
    if predictions[i] == 2:
        print("Winner: " + backup_pred_set.iloc[i, 1])
    elif predictions[i] == 1:
        print("Draw")
    elif predictions[i] == 0:
        print("Winner: " + backup_pred_set.iloc[i, 0])
    print('Probability of ' + backup_pred_set.iloc[i, 1] + ' winning: ', '%.3f'%(log_reg.predict_proba(pred_set)[i][2]))
    print('Probability of Draw: ', '%.3f'%(log_reg.predict_proba(pred_set)[i][1]))
    print('Probability of ' + backup_pred_set.iloc[i, 0] + ' winning: ', '%.3f'%(log_reg.predict_proba(pred_set)[i][0]))
    print("")
lre_score = accuracy_score(predictions, y_pred_set) * 100
print("LRE Accuracy Actual/Predict : %.2f%%" % (lre_score))

predictions = mlp.predict(pred_set)
##for i in range(fixtures.shape[0]):
##    print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
##    if predictions[i] == 2:
##        print("Winner: " + backup_pred_set.iloc[i, 1])
##    elif predictions[i] == 1:
##        print("Draw")
##    elif predictions[i] == 0:
##        print("Winner: " + backup_pred_set.iloc[i, 0])
##    print('Probability of ' + backup_pred_set.iloc[i, 1] + ' winning: ', '%.3f'%(mlp.predict_proba(pred_set)[i][2]))
##    print('Probability of Draw: ', '%.3f'%(mlp.predict_proba(pred_set)[i][1]))
##    print('Probability of ' + backup_pred_set.iloc[i, 0] + ' winning: ', '%.3f'%(mlp.predict_proba(pred_set)[i][0]))
##    print("")
mlp_score = accuracy_score(predictions, y_pred_set) * 100
print("MLP Accuracy Actual/Predict : %.2f%%" % (mlp_score))

predictions = knn.predict(pred_set)
##for i in range(fixtures.shape[0]):
##    print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
##    if predictions[i] == 2:
##        print("Winner: " + backup_pred_set.iloc[i, 1])
##    elif predictions[i] == 1:
##        print("Draw")
##    elif predictions[i] == 0:
##        print("Winner: " + backup_pred_set.iloc[i, 0])
##    print('Probability of ' + backup_pred_set.iloc[i, 1] + ' winning: ', '%.3f'%(knn.predict_proba(pred_set)[i][2]))
##    print('Probability of Draw: ', '%.3f'%(knn.predict_proba(pred_set)[i][1]))
##    print('Probability of ' + backup_pred_set.iloc[i, 0] + ' winning: ', '%.3f'%(knn.predict_proba(pred_set)[i][0]))
##    print("")
knn_score = accuracy_score(predictions, y_pred_set) * 100
print("KNN Accuracy Actual/Predict : %.2f%%" % (knn_score))

# input("press enter to exit...")

import matplotlib.pyplot as plt
import numpy as np
objects = ('Logistic Regression', 'Multi-Layer Perceptron', 'K-Neighbors-KD-Tree')
y_pos = np.arange(len(objects))
scores = [lre_score, mlp_score, knn_score]
plt.bar(y_pos, scores, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy Score')
plt.title("Comparing Model's Accuracy")
plt.show()
