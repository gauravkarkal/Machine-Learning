#cross Validation Regression R^2
from pandas import read_csv
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

#filename='housing.csv'
names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PIRATIO','B','LSTAT','MEDV']
dataframe=read_csv("housing_original.csv",delim_whitespace=True,names=names)
#dataframe=read_csv(filename,names=names)
array=dataframe.values
X=array[:,0:13]
Y=array[:,13]
#kfold=Kfold(n_splits=10,random_state=7)
model=LinearRegression()
#scporing='r2'
#results=cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
results=model.fit(X,Y)
#print(results.mean(),results.std())
print(results.coef_)
print(results.intercept_)
print(results.score(X,Y))













































































































































