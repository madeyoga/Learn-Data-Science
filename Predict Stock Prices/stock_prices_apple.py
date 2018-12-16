import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd

# plt.switch_backend('new_backend')

dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('/')[2]))
            prices.append(float(row[3]))
    return

def predict_prices(dates, prices, x):
    dates = np.reshape(dates, len(dates), -1)

    dates = dates.reshape(-1, 1)
    print(dates)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma='auto')

    svr_lin.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF Model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear Model')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    # return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

# get_data('G:\Programs\Python\Learn Data-Science\Datasets\Apple.csv')
# print(dates)
# predicted_price = predict_prices(dates, prices, 29)

df = pd.read_csv('Datasets/Apple.csv', header=None)

dates = df.index.values
prices = df.loc[:, 3]

print(df)

dates = dates.reshape(-1, 1)

print(dates)
print(prices)

svr_lin = SVR(kernel='linear', C=1e3)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

svr_lin.fit(dates, prices.ravel())
svr_rbf.fit(dates, prices.ravel())

plt.scatter(dates, prices, color='black', label='Data')
plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF Model')
plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear Model')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
