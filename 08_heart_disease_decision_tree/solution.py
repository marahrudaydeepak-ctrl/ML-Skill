import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
df=pd.read_csv('dataset.csv');X=df.drop(columns='target');y=df.target;a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42,stratify=y);m=DecisionTreeClassifier(max_depth=4,random_state=42);m.fit(a,c);p=m.predict(b);print(classification_report(d,p));print(confusion_matrix(d,p));plt.figure(figsize=(12,7));plot_tree(m,feature_names=X.columns,class_names=['No','Yes']);plt.tight_layout();plt.savefig('plots/decision_tree.svg');plt.close()
