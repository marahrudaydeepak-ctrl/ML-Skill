import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
df=pd.read_csv('dataset.csv'); X=df[['Pregnancies','Glucose','BloodPressure','BMI','Age']]; y=df.Severity
a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42,stratify=y); m=Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(max_iter=2000))]); m.fit(a,c); print(classification_report(d,m.predict(b))); print(confusion_matrix(d,m.predict(b)))
