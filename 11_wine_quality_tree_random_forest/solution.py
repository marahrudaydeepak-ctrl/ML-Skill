import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
df=pd.read_csv('dataset.csv');df['quality_class']=pd.cut(df.quality,bins=[0,5,6,10],labels=[0,1,2],include_lowest=True).astype(int);X=df.drop(columns=['quality','quality_class']);y=df.quality_class;a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
for name,m in [('Decision Tree',DecisionTreeClassifier(max_depth=5,random_state=42)),('Random Forest',RandomForestClassifier(n_estimators=200,random_state=42))]:m.fit(a,c);print(name);print(classification_report(d,m.predict(b)))
