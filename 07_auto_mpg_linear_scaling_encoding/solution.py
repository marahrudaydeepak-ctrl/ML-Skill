import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
df=pd.read_csv('dataset.csv'); X=df.drop(columns='mpg'); y=df.mpg; num=[c for c in X.columns if c!='origin']; pre=ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore'),['origin'])]);m=Pipeline([('preprocessor',pre),('regressor',LinearRegression())]);a,b,c,d=train_test_split(X,y,test_size=.2,random_state=42);m.fit(a,c);p=m.predict(b);print('RMSE',mean_squared_error(d,p)**.5,'R2',r2_score(d,p))
