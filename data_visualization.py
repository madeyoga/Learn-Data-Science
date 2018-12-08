import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

## load dataset
df_all = pd.read_csv(
    'Datasets/pml-training.csv'
    )

## count number of samples that contains (NaN)
counter_nan = df_all.isnull().sum()
counter_without_nan = counter_nan[counter_nan==0]

## remove samples
df_all = df_all[counter_without_nan.keys()]

## remove the first 7 columns which contains no discriminative information
df_all = df_all.ix[:, 7:]

columns = df_all.columns

## get x and convert it to numpy array
X = df_all.ix[:, :-1].values
x_std = StandardScaler().fit_transform(X)

## get class labels/targets y
y = df_all.ix[:, -1].values
## encode to number
class_labels = np.unique(y)
y = LabelEncoder().fit_transform(y)
print(y)

## split data into training and test set
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.1, random_state=0)

## t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
tsne = TSNE(n_components=2, random_state=0)
x_test_2d = tsne.fit_transform(x_test)

# scatter plot the sample points among 5 classes
markers=('s', 'd', 'o', '^', 'v')
color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan'}
plt.figure()
for idx, cl in enumerate(np.unique(y_test)):
    plt.scatter(x=x_test_2d[y_test==cl,0], y=x_test_2d[y_test==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show()
