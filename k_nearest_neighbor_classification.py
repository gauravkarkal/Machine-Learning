from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

dataframe=read_csv("data2.csv")
dataframe=dataframe.drop(columns=['id'],axis=1)
dataframe.dropna(inplace=True)
array=dataframe.values
Y=array[:,0]
dataframe=dataframe.drop(columns='diagnosis',axis=1)
X=dataframe.values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=8)

def accuracy():
    result=model.score(X_test,Y_test)
    print("Accuracy:",result*100.0)


def confusion():
    predicted=model.predict(X_test)
    matrix=confusion_matrix(Y_test,predicted)
    print("Confusion matrix:\n",matrix,"\n")
    
model = KNeighborsClassifier(n_neighbors=2) 
model.fit(X_train, Y_train)
accuracy()
confusion()
