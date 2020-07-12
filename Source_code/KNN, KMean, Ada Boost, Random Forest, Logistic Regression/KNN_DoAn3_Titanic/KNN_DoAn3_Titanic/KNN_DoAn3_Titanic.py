import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv('C:/Users/Administrator/Desktop/titanic/titanic/train.csv')
df_test = pd.read_csv('C:/Users/Administrator/Desktop/titanic/titanic/test.csv')
# Explore the data by printing the head of the dataset.
df_train.head()
print('Column names and data of first twenty passengers - training set:\n{}'.format(df_train.head(20)))
#Use corr function to find out the liner 
corr_matrix = df_train.corr()
print('Correlation between survived and other numerical features:\n{}'.format(corr_matrix["Survived"].sort_values(ascending = False)))
#visualize the correlation of Survived and other non numerical features.
df_train.info
#For Embarked, I have filled the NaN with port of embarkation with maximum frequency
df_train["Embarked"] = df_train ["Embarked"].fillna("S")
df_train.info
sns.barplot(
    data = df_train,
    x = 'Sex',
    y = 'Survived'
    )
plt.show()
sns.countplot(df_train['Embarked'])
plt.show()
sns.barplot (
    data = df_train,
    x = 'Pclass',
    y = 'Survived'
    )
plt.show()
#Data preparation
#features are Pclass, Sex, Fare and Embarked
q = ['PassengerId','Name','Ticket', 'Cabin', 'SibSp', 'Parch', 'Age']
df_train_set = df_train.drop(q, axis =1)
df_train_set.head()
z = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Age']
df_test_set = df_test.drop(z, axis = 1)
df_test_set.head()

#fill null Fare of test data by using mean function to get Average value
mean = df_test_set["Fare"].mean()
df_test_set["Fare"] = df_test_set["Fare"].fillna(mean)
df_test_set.info()

#Then translate non-numerical features to numerical features.
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df_train_set.iloc [:,2] = labelencoder.fit_transform(df_train_set.iloc[:,2].values)
df_train_set.iloc [:,4] = labelencoder.fit_transform(df_train_set.iloc[:,4].values)

df_test_set.iloc [:,2] = labelencoder.fit_transform(df_test_set.iloc[:,2].values)
df_test_set.iloc [:,4] = labelencoder.fit_transform(df_test_set.iloc[:,4].values)

df_train_set.info()

#Split train data into X_train, X_test, y_train, y_test.
X = df_train_set.iloc[:, 1:5].values
Y = df_train_set.iloc[:, 0].values

#As test_size = 0.4 the function will assign 40% of the data to the test subsets and the remaining 60% to the train subsets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X,Y,test_size= 0.4, random_state = 4)

print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

#k-nearest neighbors (KNN) model
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit (X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append (metrics.accuracy_score(y_test, y_pred))
print(scores)



import matplotlib.pyplot as plt
# Data for plotting
fig, ax = plt.subplots()
ax.plot(k_range, scores)
ax.set(xlabel='Value of K for KNN', ylabel='Testing Accuracy',
       title='Testing Accuracy of k value')
fig.savefig("test.png")
plt.show()

#K value equal 3 has the highest accuracy rate.
knn = KNeighborsClassifier (n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Accuracy of k-NN test: {}'.format(metrics.accuracy_score(y_test, y_pred)))

#Finally, we can use this model to make predictions on out of sample data. And generate a new data frame ‘submission’.
df_test_set.info()
test = df_test_set.iloc[:,1:5].values
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit (X,Y)
Y_pred = knn.predict(test)

knn_submission = pd.DataFrame({
    "PassengerId" : df_test_set["PassengerId"],
    "Survived" : Y_pred
    })
#ẽport to csv
knn_submission.to_csv("C:/Users/Administrator/Desktop/titanic/titanic/knn_submission.csv", encoding='utf-8')



#Logistic Regression
from sklearn.linear_model import LogisticRegression


# Instantiate our model
logreg = LogisticRegression()
# Fit our model to the training data
logreg.fit(X_train, y_train)
# Predict on the test data
logreg_predictions = logreg.predict(X_test)
print('Accuracy of Logistic Regression test: {}'.format(metrics.accuracy_score(y_test, logreg_predictions)))


df_test_set.info()
test = df_test_set.iloc[:,1:5].values
# Instantiate our model
logreg = LogisticRegression()
logreg.fit (X,Y)
logreg_pred = logreg.predict(test)
logreg_submission = pd.DataFrame({
    "PassengerId" : df_test_set["PassengerId"],
    "Survived" : logreg_pred
    })
logreg_submission.to_csv('C:/Users/Administrator/Desktop/titanic/titanic/LogisticRegression_SS_OH_FE2.csv')

#Adaboost
from sklearn.ensemble import  AdaBoostClassifier

# Instantiate our model
adaboost = AdaBoostClassifier()
# Fit our model to the training data
adaboost.fit(X_train, y_train)
# Predict on the test data
adaboost_predictions = adaboost.predict(X_test)
print('Accuracy of adabboost test: {}'.format(metrics.accuracy_score(y_test, adaboost_predictions)))


df_test_set.info()
test = df_test_set.iloc[:,1:5].values
adaboost = AdaBoostClassifier()
adaboost.fit (X,Y)
adaBoost_pred = adaboost.predict(test)
adaboost_submission = pd.DataFrame({
    "PassengerId" : df_test_set["PassengerId"],
    "Survived" : adaBoost_pred
    })
adaboost_submission.to_csv(
    'C:/Users/Administrator/Desktop/titanic/titanic/AdaptiveBoosting_SS_OH_FE.csv')

#random forest 
from sklearn.ensemble import  RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
random_forest_predictions = random_forest.predict(X_test)
print('Accuracy of Random Forest test: {}'.format(
    metrics.accuracy_score(y_test, random_forest_predictions)))


df_test_set.info()
test = df_test_set.iloc[:,1:5].values
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit (X,Y)
random_forest_pred = random_forest.predict(test)
random_forest_submission = pd.DataFrame({
    "PassengerId" : df_test_set["PassengerId"],
    "Survived" : random_forest_pred
    })
random_forest_submission.to_csv(
    'C:/Users/Administrator/Desktop/titanic/titanic/RandomForest_SS_OH.csv')

#kmean
from sklearn import datasets, cluster
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

kmeans = cluster.KMeans(n_clusters=2) # You want cluster the passenger records into 2: Survived or Not survived
kmeans.fit(X)
cluster.KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == Y[i]:
        correct += 1

print('Accuracy of K mean test: {}'.format(correct/len(X)));