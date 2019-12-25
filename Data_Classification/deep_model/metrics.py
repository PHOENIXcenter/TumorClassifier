from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np


def stat():
	y_test = np.load('tmp/y_test.npy')
	y_pred = np.load('tmp/y_pred.npy')
	y_scores = np.load('tmp/y_scores.npy')
	acc = accuracy_score(y_pred,y_test)
	f1 = f1_score(y_test, y_pred)
	auc = roc_auc_score(y_test,y_scores)
	print('Test metrics:')
	print('ACC:', acc, ' F1-Score:', f1, ' AUC:', auc)


if __name__ == '__main__':
	stat()
