from matplotlib.colors import ListedColormap

def plot_decision_regions(x,y,classifiler,resolution=0.02):
    
    marker=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))]#labelの数分だけ

    x1_min,x1_max=x[:,0].min()-1,x[:0].max()+1
