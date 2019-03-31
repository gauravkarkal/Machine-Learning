from pandas import set_option
#import matplotlib.pyplot as plt
from pandas import read_csv
#from pandas import *
names=['preg','plas','pres','skin','test','mass','pedi','age','binary']
data=read_csv("pima-indians-diabetes.data.csv",names=names)
set_option('precision',3)
correlation=data.corr(method='pearson')
print(correlation)
data.plot(kind='box',subplots=True,layout=(3,3))

from pandas.tools.plotting import scatter_matrix
scatter_matrix(data)
#plt.scatter(data)