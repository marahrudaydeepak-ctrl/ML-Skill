import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
df=pd.read_csv('dataset.csv');X=df.drop(columns='target');y=df.target;a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42,stratify=y);m=RandomForestClassifier(n_estimators=300,max_depth=8,random_state=42);m.fit(a,c);print(classification_report(d,m.predict(b)));print(pd.Series(m.feature_importances_,index=X.columns).sort_values(ascending=False))
