from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
import numpy as np
import os


args = {
'svm_args':{'C':0.1, 'kernel':'linear'}
'lr_args':{'penalty':'l2'}
'ppn_args':{'max_iter':100, 'eta0':0.0001, 'random_state':0}
'rf_args':{'n_estimators':100,'max_depth':20,'max_features':0.2}
'gdbt_args':{'n_estimators':100,'learning_rate':0.1}
}


def load_data():
	X_train = np.load("x_train.npy")
	y_train = np.load("y_train.npy")
	X_test = np.load("x_test.npy")
	y_test = np.load("y_test.npy")
	return X_train, y_train, X_test, y_test


def svm_classify(X_train,y_train,X_test,y_test):
	clf = SVC(C=0.1, kernel='linear')
	#clf = SVC(**args['svm_args'])
	clf.fit(X_train,y_train)
	svm_y_pred = clf.predict(X_test)
	acc = accuracy_score(svm_y_pred,y_test)
	f1 = f1_score(y_test, svm_y_pred)
	y_scores = clf.decision_function(X_test)
	auc = roc_auc_score(y_test,y_scores)
	strs = "SVM Test_acc: {:.6f}".format(acc)
	strs = strs + " F1-score: {:.6f}".format(f1)
	strs = strs + " AUC: {:.6f}".format(auc)
	print(strs)


def lr_classify(X_train,y_train,X_test,y_test):
	clf = LogisticRegression(penalty='l2')
	#clf = LogisticRegression(**args['lr_args'])
	clf.fit(X_train,y_train)
	clf_y_pred = clf.predict(X_test)
	acc = accuracy_score(clf_y_pred,y_test)
	f1 = f1_score(y_test, clf_y_pred)
	y_scores = clf.decision_function(X_test)
	auc = roc_auc_score(y_test,y_scores)
	strs = "LR Test_acc: {:.6f}".format(acc)
	strs = strs + " F1-score: {:.6f}".format(f1)
	strs = strs + " AUC: {:.6f}".format(auc)
	print(strs)


def ppn_classify(X_train,y_train,X_test,y_test):
	ppn = Perceptron(max_iter=100, eta0=0.0001, random_state=0)
	#ppn = Perceptron(**args['ppn_args'])
	ppn.fit(X_train,y_train)
	ppn_y_pred = ppn.predict(X_test)
	acc = accuracy_score(ppn_y_pred,y_test)
	f1 = f1_score(y_test, ppn_y_pred)
	y_scores = ppn.decision_function(X_test)
	auc = roc_auc_score(y_test,y_scores)
	strs = "PPN Test_acc: {:.6f}".format(acc)
	strs = strs + " F1-score: {:.6f}".format(f1)
	strs = strs + " AUC: {:.6f}".format(auc)
	print(strs)


def rf_classify(X_train,y_train,X_test,y_test):
	rfc = RandomForestClassifier(n_estimators = 100,max_depth = 20,max_features=0.2)
	#rfc = RandomForestClassifier(**args['rf_args'])
	rfc.fit(X_train,y_train)
	rf_y_pred = rfc.predict(X_test)
	acc = accuracy_score(rf_y_pred,y_test)
	f1 = f1_score(y_test, rf_y_pred)
	y_scores = rfc.predict_proba(X_test)[:,1]
	auc = roc_auc_score(y_test,y_scores)
	strs = "RF Test_acc: {:.6f}".format(acc)
	strs = strs + " F1-score: {:.6f}".format(f1)
	strs = strs + " AUC: {:.6f}".format(auc)
	print(strs)


def gdbt_classify(X_train,y_train,X_test,y_test):
	gbc = GradientBoostingClassifier(n_estimators = 100,learning_rate=0.1)
	#gbc = GradientBoostingClassifier(**args['gdbt_args'])		
	gbc.fit(X_train,y_train)
	gb_y_pred = gbc.predict(X_test)
	acc = accuracy_score(gb_y_pred,y_test)
	f1 = f1_score(y_test, gb_y_pred)
	y_scores = gbc.predict_proba(X_test)[:,1]
	auc = roc_auc_score(y_test,y_scores)
	strs = "GDBT Test_acc: {:.6f}".format(acc)
	strs = strs + " F1-score: {:.6f}".format(f1)
	strs = strs + " AUC: {:.6f}".format(auc)
	print(strs)


def classify(X_train,y_train,X_test,y_test):
	svm_classify(X_train,y_train,X_test,y_test)
	lr_classify(X_train,y_train,X_test,y_test)
	ppn_classify(X_train,y_train,X_test,y_test)
	rf_classify(X_train,y_train,X_test,y_test)
	gdbt_classify(X_train,y_train,X_test,y_test)


if __name__ == '__main__':
	X_train, y_train, X_test, y_test = load_data()
	classify(X_train,y_train,X_test,y_test)

