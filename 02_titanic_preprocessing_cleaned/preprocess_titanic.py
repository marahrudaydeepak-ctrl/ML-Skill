from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
df=sns.load_dataset('titanic').drop(columns=['alive','class','deck'],errors='ignore')
Path('data').mkdir(exist_ok=True)
clean=df.copy(); clean['age']=clean.age.fillna(clean.age.median()); clean['fare']=clean.fare.fillna(clean.fare.median()); clean['embarked']=clean.embarked.fillna(clean.embarked.mode()[0]); clean.to_csv('data/titanic_cleaned.csv',index=False)
X=df.drop(columns='survived'); y=df.survived; num=X.select_dtypes(include='number').columns; cat=X.select_dtypes(exclude='number').columns
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
A=pre.fit_transform(Xtr); print('Saved cleaned dataset; transformed train shape:',A.shape)
