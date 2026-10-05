import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
df=pd.read_csv("dataset.csv")
X=df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
y=df["Survived"]
pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),["Pclass","Age","SibSp","Parch","Fare"]),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),["Sex","Embarked"])])
m=Pipeline([("preprocessor",pre),("classifier",LogisticRegression(max_iter=1000))])
a,b,c,d=train_test_split(X,y,test_size=.2,random_state=42,stratify=y);m.fit(a,c);print(classification_report(d,m.predict(b)))
