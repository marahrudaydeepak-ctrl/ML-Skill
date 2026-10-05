import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

df=pd.read_csv('dataset.csv'); X=df.drop(columns='Survived'); y=df.Survived
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),['Pclass','Age','SibSp','Parch','Fare']),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),['Sex','Embarked'])])
m=Pipeline([('preprocessor',pre),('classifier',LogisticRegression(max_iter=1000))]); a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42,stratify=y); m.fit(a,c); print(classification_report(d,m.predict(b))); print(confusion_matrix(d,m.predict(b)))
