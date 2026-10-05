import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
df=pd.read_csv("dataset.csv"); X=df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),["Pclass","Age","SibSp","Parch","Fare"]),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),["Sex","Embarked"])])
z=pre.fit_transform(X); cols=["Pclass","Age","SibSp","Parch","Fare"]+list(pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(["Sex","Embarked"]));pd.DataFrame(z,columns=cols).to_csv("cleaned_dataset.csv",index=False);print(z.shape)
