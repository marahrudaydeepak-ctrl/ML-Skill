import numpy as np,pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
df=fetch_openml('autoMpg',version=1,as_frame=True).frame; y=pd.to_numeric(df.pop('mpg'),errors='coerce'); X=df
for c in X.select_dtypes(exclude='number').columns:
    if X[c].nunique()>20: X=X.drop(columns=c)
mask=y.notna(); X=X.loc[mask]; y=y.loc[mask]; num=X.select_dtypes(include='number').columns; cat=X.select_dtypes(exclude='number').columns
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42); p=Pipeline([('pre',pre),('model',LinearRegression())]); p.fit(Xtr,ytr); pred=p.predict(Xte)
print('MAE',mean_absolute_error(yte,pred),'RMSE',np.sqrt(mean_squared_error(yte,pred)),'R2',r2_score(yte,pred))
