from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
names=['preg','plas','pres','skin','test','mass','pedi','age','binary']
dataframe=read_csv("pima-indians-diabetes.data.csv",names=names)
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]

scaler=MinMaxScaler(feature_range=(0,1))
scaler=StandardScaler().fit(X)
rescaledX=scaler.fit_transform(X)
set_printoptions(precision=3)
print(rescaledX[0:5,:])