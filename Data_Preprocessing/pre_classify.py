from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import sys
import os


p = int(sys.argv[1]) #100
s = int(sys.argv[2]) #10
up = int(sys.argv[3]) #10
l = (np.load('Exper_data/con_keys.npy').shape[0])*2


def Binarization(x):
	a,b = np.percentile(x[x>0], [0.1,50])
	x[x>b] = 1
	x[x<a] = 0
	t1 = x>a
	t2 = x<b
	tt = t1&t2
	ind = np.where(tt == True)
	x[ind] = 0.75
	return x


def get_data(path,num):
	files = os.listdir(path)
	nn = len(files)
	Xt = np.zeros((nn, l))
	for m,n in enumerate(files):
		ms = np.load(path+"/"+n)
		ms = Binarization(ms).tolist()
		Xt[m,:] = ms
	labels = [num]*nn
	return Xt,labels


def load_data():
	X0,labels_t0 = get_data('Train/Fea-Good', 0)
	X1,labels_t1 = get_data('Train/Fea-Bad', 1)
	Xt = np.array([j for j in X0]+[k for k in X1])
	labels_t = np.array(labels_t0 + labels_t1)
	rule = pre_classify(Xt, labels_t)
	X2,labels_s0 = get_data('Test/Fea-Good', 0)
	X3,labels_s1 = get_data('Test/Fea-Bad', 1)
	Xs = np.array([j for j in X2]+[k for k in X3])
	labels_s = np.array(labels_s0 + labels_s1)
	return Xt, Xs, labels_t, labels_s, rule


def pre_classify(Xt, labels):
	k = int((l-p)/s)+1
	dic = {}
	for i in range(k):
		xp = Xt[:,i*s:i+p]
		accs = []
		kf = KFold(n_splits=10)
		for tr, ts in kf.split(xp):
			xr = Xt[tr,:]
			yr = labels[tr]
			xs = Xt[ts,:]
			ys = labels[ts]
			clf = SVC()
			clf.fit(xr, yr)
			svm_y_pred = clf.predict(xs)
			acc_ = accuracy_score(svm_y_pred,ys)
			accs.append(acc_)
		acc = np.mean(accs)
		dic[i] = acc
	x = list(dic.values())
	_, me = np.percentile(x, [0, up])
	r = []
	for i in dic:
		if dic[i] >= me:
			r.append(i)
	return r


def save_per_data(Xt, Xs, y_t, y_s, r):
	X_ = np.zeros((len(Xt), len(r)*p))
	for j,i in enumerate(r):
		xpt = Xt[:, i*s:i+p]
		X_[:,j*p:(j+1)*p] = xpt
	X_train = X_
	X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, y_t, stratify = y_t, test_size=0.1, random_state = 123)
	np.save('Exper_data/x_train.npy', X_tr)
	np.save('Exper_data/y_train.npy', lab_tr)
	np.save('Exper_data/x_vld.npy', X_vld)
	np.save('Exper_data/y_vld.npy', lab_vld)
	X_ = np.zeros((len(Xs), len(r)*p))
	for j,i in enumerate(r):
		xps = Xs[:, i*s:i+p]
		X_[:,j*p:(j+1)*p] = xps
	X_test = X_
	np.save('Exper_data/x_test.npy',X_test)
	np.save('Exper_data/y_test.npy',y_s)


if __name__ == '__main__':
	Xt, Xs, labels_t, labels_s, rule = load_data()
	save_per_data(Xt, Xs, labels_t, labels_s, rule)
