from SGD import AdalineSGD
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#generate datasets
df=pd.read_csv('iris.csv',header=None)
#print(df.tail())

y=df.iloc[:100,4].values
y=np.where(y=='Iris-versicolor',1,-1)
x=df.iloc[:100,[0,2]].values

#generate AdalineSGD
ppn=AdalineSGD(eta=0.01,n_iter=15)
ppn.fit(x,y)

#generate figure
fig=plt.figure()
plane=fig.add_subplot(1,1,1)
plane.plot(range(1,len(ppn.cost_)+1),ppn.cost_)

plane.set_xlabel('epocs')
plane.set_ylabel('costs')


fig.show()
plt.show()
