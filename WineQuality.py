import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

lr = LogisticRegression(random_state=1)
nb = MultinomialNB()
dt = DecisionTreeClassifier(random_state = 0)
gbn = GradientBoostingClassifier(n_estimators = 10)
rf = RandomForestClassifier(random_state=2)



df = pd.read_csv('C:/Users/parab/OneDrive/Desktop/p3/Dataset4.csv')
print(df)
df.head()
print(df)
print(df.isnull().sum())

df['fixed acidity'].fillna((df['fixed acidity']).mean(), inplace = True)
df['pH'].fillna((df['pH']).mean(), inplace = True)

'''sns.boxplot(df['fixed acidity'])
plt.show()


print(df['fixed acidity'])
Q1 = df['fixed acidity'].quantile(0.25)
Q3 = df['fixed acidity'].quantile(0.75)

IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 + 1.5*IQR

print(upper)
print(lower)

sns.boxplot(df['fixed acidity'])
plt.show()'''

print(df.isnull().sum())
X = df.drop('fixed acidity', axis = 1)
X = X.drop('volatile acidity', axis =1)
X = X.drop('citric acid', axis =1)
X = X.drop('residual sugar', axis =1)
X = X.drop('chlorides', axis =1)
X = X.drop('free sulfur dioxide', axis =1)
X = X.drop('total sulfur dioxide', axis =1)
X = X.drop('density', axis =1)
X = X.drop('sulphates', axis =1)

Y = df['quality']
print(X.isnull().sum())

df['pH'].fillna((df['pH']).mean(), inplace = True)
print(X.isnull().sum())


le = LabelEncoder()
le.fit(X['type'])
(X['type'])=le.transform(X['type'])
print((X['type']))

x=X.drop('pH',axis=1)
x=X.drop('quality',axis=1)
y=df['quality']

bestFeatures = SelectKBest(score_func=chi2, k = 'all')
fit = bestFeatures.fit(x,y)                                  #training our model
dfscores = pd.DataFrame(fit.scores_)                         #storing scores
dfcolumns = pd.DataFrame(x.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', ' Score']

print(featureScores)




X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=2,test_size = 0.2)

#Random Forest Classifier
rf.fit(X_train, Y_train)
y_pred=rf.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy :", accuracy)


#Logistic Regression
lr.fit(X_train, Y_train)
y_pred=rf.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy :", accuracy)

#MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, Y_train)
y_pred = nb.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy NB:", accuracy)








