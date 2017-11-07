import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import os 


def main():
    if 'wine.csv' in os.getcwd():
        df_wine=pd.read_csv('wine.csv')
    else:
        df_wine=pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
        df_wine.to_csv('wine.csv',index=False)

    df_wine.columns=['Class label','Alocohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
    x,y=df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


if __name__=='__main__':
    main()

