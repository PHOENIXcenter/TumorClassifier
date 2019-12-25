import numpy as np

nums = 100
cover = 500


def augment(X_train, y_train, arr):
	tmpx = []
	tmpy = []
	for i in arr:
		x = X_train[i]
		inds = np.random.randint(0,len(x),cover)
		inds = np.array(list(set(inds)))		
		x[inds] = 0
		tmpx.append(x)
		tmpy.append(y_train[i])
	return tmpx, tmpy


def generate_data()
	X_train = np.load("x_train.npy")
	y_train = np.load("y_train.npy")
	arr = np.random.randint(0,len(X_train),nums)
	x_add, y_add = augment(X_train, y_train, arr)
	X_train = np.concatenate((x_add,X_train),axis=0)
	y_train = np.concatenate((y_add,y_train),axis=0)
	return X_train, y_train


def reshape(x):
	n = len(x)
	return x.reshape(n,256,16)


def trans_size():
	X_train, y_train = generate_data()
	X_train = reshape(X_train)
	X_test = np.load("x_test.npy")
	y_test = np.load("y_test.npy")
	X_test = reshape(X_test)
	np.save("x_train.npy", X_train)
	np.save("y_train.npy", y_train)
	np.save("x_test.npy", X_test)
	np.save("y_test.npy", y_test)


if __name__ == '__main__':
	trans_size()

