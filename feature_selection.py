
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("RR.csv")
rr_new = pd.read_csv("new.csv")
rr = df[df['round'] != 'warmup']

X = rr_new.values[:,1:7]
Y = rr_new.values[:,7]

model = LogisticRegression()
# Number of features
rfe = RFE(model, 1)
fit = rfe.fit(X, Y)
print("Num Features: %d"% fit.n_features_)
print("Selected Features: %s"% fit.support_)
print("Feature Ranking: %s"% fit.ranking_)

#---------------------------------------------Decision Tree Predictor Code-----------------------------------------------

# from sklearn.cross_validation import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# from sklearn import tree
# acceleration = []
# jerk = []
# costheta = []
# labels = []
# for i in rr.ID.unique():
#     acceleration.append(rr[rr.ID == i]["acc"].mean())
#     jerk.append(rr[rr.ID == i]["jerk"].mean())
#     costheta.append(rr[rr.ID == i]["costheta"].mean())
#     labels.append(rr[rr.ID == i]["mabc_binary_score"].mean())
# X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
# dtree = DecisionTreeClassifier()
# dtree.fit(X_train, Y_train)
# y_pred = dtree.predict(X_test)
# print(y_pred)
# print(accuracy_score(Y_test,y_pred)*100)

#-----------------------------------------------------------------------------------------------------------------------