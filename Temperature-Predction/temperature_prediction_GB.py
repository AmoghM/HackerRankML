'''
Accuracy:- 62.82%
Question Link:- https://www.hackerrank.com/challenges/temperature-predictions/problem

Original data shape:
Input (X): [1,2,3]
Temperature (Y): [5.,7.3,6.2]
Prediction (X'): [4]

Example for reshaped data format for fit():
Input (X) is [[1],
              [2],
              [3]
             ]
Temperature (Y): [5.,7.3,6.2]

Prediction  (X') s [[4]]
'''

import numpy as np
from sklearn import ensemble

def get_data(n,month):
    """Data formatting

    Args:
        n: Total rows of dataset..
        month: a month's data instance
    """

    t_min = []
    t_max = []
    x = []
    for i in range(0,int(n)):
        row = input().split("\t") # Data format: year    month    max_tem min_tem (tab-separated)
        x.append(month[row[1]]) #month
        try:
            t_max.append(float(row[2])) #Append the maximum temperature. If temperature is missing then raise ValueError.
        except ValueError:
            ans = interpolate(t_max,x[:-1],[x[-1]]) #Get interpolated data.
            t_max.append(ans) #Append the predicted value in place of missing value.
        try:
            t_min.append(float(row[3])) #Append the minimum temperature. If temperature is missing then raise ValueError.
        except ValueError:
            ans = interpolate(t_min,x[:-1],[x[-1]]) #Get interpolated data.
            t_min.append(ans) #Append the predicted value in place of missing value.

def interpolate(temp,input,pred):
    """Evaluates interploated value for the missing temperature

    Args:
        temp (list): Max/Min temperature records
        input (list): Month names
        pred (list): Missing temperature month name

    Returns: interpolated value
    """

    input = np.reshape(input,(-1,1)) #convert shape from 1-D array to 2D array with 1 column per row.
    pred = np.reshape(pred,(1,-1)) #convert shape from 1-D array to 2D array with 1 column per row.
    params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
    model = ensemble.GradientBoostingRegressor(**params) #Initializing GBR
    model.fit(input, temp) 
    ans = model.predict(pred)[0]
    print(ans)
    return ans

if __name__=='__main__':
    month = {'January':1,'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8,'September':9,'October':10,'November':11,'December':12}
    n = input()
    header = input()
    get_data(n,month)