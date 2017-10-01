import pandas as pd

df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
#print(df[40:110])
df.to_csv('iris.csv',header=None,index=False)
