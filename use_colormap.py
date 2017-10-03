from matplotlib.colors import ListedColormap

def plot_decision_regions(x,y,classifiler,resolution=0.02):
    
    marker=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))]#labelの数分だけ

    x1_min,x1_max=x[:,0].min()-1,x[:0].max()+1
    x2_min,x2_max=x[:,1].min()-1,x[:1].max()+1
    
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
