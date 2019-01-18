
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("RR.csv")
rr_new = pd.read_csv("new_all.csv")
rr_new_update = pd.read_csv("new_update.csv")
rr = df[df['round'] != 'warmup']
# Features and target data of dataset means
X = rr_new.values[:,1:7]
Y = rr_new.values[:,7]

# Features and target data of dataset all
X_new2 = rr_new.values[:,1:7]
Y_new2 = rr_new.values[:,7]

# Features and target data of dataset all update after Feature selection
X_new_update = rr_new_update.values[:,1:4]
Y_new_update = rr_new_update.values[:,4]
#------------------------------------------------------Feature Selection--------------------------------------------------

model = LogisticRegression()
# Number of features
rfe = RFE(model, 1)
fit = rfe.fit(X_new2, Y_new2)
print("Num Features: %d"% fit.n_features_)
print("Selected Features: %s"% fit.support_)
print("Feature Ranking: %s"% fit.ranking_)


from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#-----------------------------------------------------------Means---------------------------------------------------------

acceleration = []
jerk = []
costheta = []
labels = []
for i in rr.ID.unique():
    acceleration.append(rr[rr.ID == i]["acc"].mean())
    jerk.append(rr[rr.ID == i]["jerk"].mean())
    costheta.append(rr[rr.ID == i]["costheta"].mean())
    labels.append(rr[rr.ID == i]["mabc_binary_score"].mean())

#---------------------------------------------Decision Tree Predictor Code-----------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split( X_new2, Y_new2)
dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)
y_pred = dtree.predict(X_test)
acc = accuracy_score(Y_test,y_pred)
print(acc*100)

X_train2, X_test2, Y_train2, Y_test2 = train_test_split( X_new_update, Y_new_update)
dtree2 = DecisionTreeClassifier()
dtree2.fit(X_train2, Y_train2)
y_pred2 = dtree2.predict(X_test2)
acc2 = accuracy_score(Y_test2,y_pred2)
print(acc2*100)
print('percentage increase:')
print((acc2/acc*100)-100)