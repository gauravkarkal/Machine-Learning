from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
'''
#Diabetes
names=['preg','plas','pres','skin','test','mass','pedi','age','binary']
dataframe=read_csv("pima-indians-diabetes.data.csv",names=names)
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]
'''
enc=OneHotEncoder()

#Titanic
dataframe=read_csv("k5_train.csv")
dataframe=dataframe.drop(columns=['Resolution','Address','Descript'],axis=1)
array=dataframe.values
Y=array[:,1]
dataframe=dataframe.drop(columns='PdDistrict',axis=1)
X=dataframe.values
X=X[:1000,:]
Y=Y[:1000]
enc.fit(X)
enc.fit(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=8)

def accuracy():
    result=model.score(X_test,Y_test)
    print("Accuracy:",result*100.0)


def confusion():
    predicted=model.predict(X_test)
    matrix=confusion_matrix(Y_test,predicted)
    print("Confusion matrix:\n",matrix,"\n")
    
'''
print("Logistic Regression")
model=LogisticRegression()
model.fit(X_train,Y_train)
accuracy()
confusion()
'''
print("SVC")
model=SVC(kernel='linear',gamma=1)
model.fit(X_train,Y_train)
accuracy()
confusion()
'''
print("Random Forest")
model=RandomForestClassifier(n_estimators=4,max_features=3)
model.fit(X_train,Y_train)
accuracy()
confusion()

print("Decision Tree")
model=DecisionTreeClassifier()
model.fit(X_train,Y_train)
accuracy()
confusion()

print("KNN")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, Y_train)
accuracy()
confusion()

print("Gaussian_NB")
model=GaussianNB()
model.fit(X_train, Y_train)
accuracy()
confusion()
'''