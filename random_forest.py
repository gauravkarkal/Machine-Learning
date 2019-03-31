from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#from sklearn.tree import export_graphviz
#export_graph(tree_in_forest,feature_name=X.coloumns,filled=True,rounded=True)
names=['preg','plas','pres','skin','test','mass','pedi','age','binary']
dataframe=read_csv("pima-indians-diabetes.data.csv",names=names)
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]

num_trees=4
max_features=3


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=7)

model=RandomForestClassifier(n_estimators=num_trees,max_features=max_features)
print(type(model))

model.fit(X_train,Y_train)
predicted=model.predict(X_test)
matrix=confusion_matrix(Y_test,predicted)
print(matrix)
print(model.feature_importances_)
print(model.predict([[8,183,64,0,0,23.3,0.672,32]]))
print(model.predict_proba([[8,183,64,0,0,23.3,0.672,32]]))
print(model.predict_proba(X_test))
a=model.estimators_[3]
print(a)