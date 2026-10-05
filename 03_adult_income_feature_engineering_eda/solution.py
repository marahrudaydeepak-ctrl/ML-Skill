import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("dataset.csv");df["is_senior"]=(df["age"]>=50).astype(int);df["weekly_income_proxy"]=df["hours_per_week"]*df["capital_gain"];df["income_binary"]=(df["income"]==">50K").astype(int)
X=df[["age","hours_per_week","capital_gain","is_senior","workclass","education"]];y=df["income_binary"]
pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),["age","hours_per_week","capital_gain","is_senior"]),("cat",OneHotEncoder(handle_unknown="ignore"),["workclass","education"])])
m=Pipeline([("preprocessor",pre),("classifier",LogisticRegression(max_iter=1000))]);a,b,c,d=train_test_split(X,y,test_size=.2,random_state=42,stratify=y);m.fit(a,c);print(m.score(b,d))
