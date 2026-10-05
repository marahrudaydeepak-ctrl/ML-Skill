import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
df=fetch_openml('adult',version=2,as_frame=True).frame
print(df.head()); print(df.shape); print(df.isna().sum())
df['capital_net_gain']=df['capital-gain']-df['capital-loss']; df['has_capital_gain']=(df['capital-gain']>0).astype(int); df['age_band']=pd.cut(df.age,[0,25,40,60,100],labels=['young','adult','midlife','senior']); df['overtime_hours']=(df['hours-per-week']>40).astype(int)
target='class' if 'class' in df else 'income'; y=df.pop(target).astype(str).str.strip().str.replace('.','',regex=False); X=df
num=X.select_dtypes(include='number').columns; cat=X.select_dtypes(exclude='number').columns
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
p=Pipeline([('pre',pre),('model',LogisticRegression(max_iter=1000))]); p.fit(Xtr,ytr); print(classification_report(yte,p.predict(Xte)))
