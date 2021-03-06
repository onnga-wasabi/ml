import numpy as np
class AdalineSGD:
    def __init__(self,eta,n_iter):
        self.eta=eta
        self.n_iter=n_iter

    def fit(self,x,y):
        """xは標本と特徴の2次元配列
        yは正解のラベル
        """
        self.w_=np.zeros(1+x.shape[1])
        self.cost_=[]

        for i in range(self.n_iter):
            cost=0
            s=np.random.permutation(len(y))
            x,y=x[s],y[s]
            for xi,yi in zip(x,y):
                cost+=self.update_weights(xi,yi)
            avg_cost=cost/len(y)
            self.cost_.append(avg_cost)
        return self

    def net_input(self,x):
        return np.dot(x,self.w_[1:])+self.w_[0]
    
    def predict(self,x):   
        return np.where(self.net_input(x)>=0.0,1,-1)

    def update_weights(self,xi,yi):
        output=self.net_input(xi)
        error=yi-output
        self.w_[1:]+=self.eta*xi.dot(error)
        self.w_[0]+=self.eta*error
        cost=0.5*error**2
        return cost

    def accuracy(self,x,y):
        y_pred=self.predict(x)
        a=(len(y)-(y_pred!=y).sum())/len(y)
        return self
