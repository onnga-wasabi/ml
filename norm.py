import numpy as np
class AdalineGD:
    def __init__(self,eta,n_iter):
        self.eta=eta
        self.n_iter=n_iter

    def fit(self,x,y):
        """xは標本と特徴の2次元配列
        yは正解のラベル
        """
        self.w_=np.zeros(1+x.shape[1])
        self.cost_=[]

        x_std=np.copy(x)
        x_std[:,0]=(x[:,0]-x[:,0].mean())/x.std()#0列目の要素の全てに対しての平均と標準偏差
        x_std[:,1]=(x[:,1]-x[:,1].mean())/x.std()#1列目の要素の全てに対しての平均と標準偏差


        for i in range(self.n_iter):#iでイテレートしていることに注意
            output=self.net_input(x_std)
            errors=y-output
            self.w_[1:]+=self.eta*x_std.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
            cost=(errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def net_input(self,x):
        return np.dot(x,self.w_[1:])+self.w_[0]
    
    def predict(self,x):   
        return np.where(self.net_input(x)>=0.0,1,-1)
