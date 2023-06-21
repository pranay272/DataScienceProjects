import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

df=pd.read_csv("C:/Users/parab/OneDrive/Desktop/p3/Iris.csv")


sns.boxplot(df['SepalLengthCm'])
plt.show()

#Dealing with outliers using Interquantile range

print(df['SepalLengthCm'])
Q1 = df['SepalLengthCm'].quantile(0.25)
Q3 = df['SepalLengthCm'].quantile(0.75)

IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 + 1.5*IQR

print(upper)
print(lower)

out1 = df[df['SepalLengthCm'] < lower].values
out2 = df[df['SepalLengthCm'] > upper].values

df['SepalLengthCm'].replace(out1,lower,inplace=True)
df['SepalLengthCm'].replace(out2,upper,inplace=True)

print(df['SepalLengthCm'])


# Principal Component Analysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df['SepalLengthCm'].fillna(df['SepalLengthCm'].mean(), inplace = True)
df['SepalWidthCm'].fillna(df['SepalWidthCm'].mean(), inplace = True)
df['PetalLengthCm'].fillna(df['PetalLengthCm'].mean(), inplace = True)


# Preparing x and y

x=df.drop('Id',axis=1)
x=x.drop('Species',axis=1)
y=df['Species']
# print(x)
# print(y)

bestFeatures = SelectKBest(score_func=chi2, k = 'all')
fit = bestFeatures.fit(x,y)                                  #training our model
dfscores = pd.DataFrame(fit.scores_)                         #storing scores
dfcolumns = pd.DataFrame(x.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', ' Score']

print(featureScores)

logr = LogisticRegression()
pca  = PCA(n_components = 2) #n_components refers to redusing the features (no. of features redused)

x = df.drop('Id', axis = 1)
x = x.drop('Species', axis =1)
y = df['Species']

pca.fit(x)
x = pca.transform(x)

print(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0,test_size = 0.3)

logr.fit(x_train,y_train)

y_pred=logr.predict(x_test)
print(accuracy_score(y_test,y_pred))

