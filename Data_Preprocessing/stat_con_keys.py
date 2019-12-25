import numpy as np
import os
import sys


n1 = int(sys.argv[1]) #0.2
n2 = int(sys.argv[2]) #0.2
l = int(sys.argv[3]) #15
h = int(sys.argv[4]) #95


# 对每一类样本的公共质合比进行统计
def stat_every(path,low,high,w):
	files = os.listdir(w+'/'+path[0])
	tt = 0
	arr = []
	for f in files:
		vv = []
		f = np.load(w+'/'+path[0]+'/'+f)
		#print(f.shape)
		x1 = f[:,1]
		x2 = f[:,2]
		# 筛掉极大和极小的强度值 为了不让极端值影响质合比计数
		me1,me2 = np.percentile(x1, [low, high])
		for i in range(len(x1)):
			if x1[i] > me1 and x1[i] < me2:
				vv.append(x2[i])
		arr.append(list(set(vv)))
	dic = {}
	for k in arr:
		for j in k:
			if j in dic.keys():
				dic[j] += 1
			else:
				dic[j] = 1
	ll = []
	dn = path[1]*len(files)
	for i in dic.keys():
		if dic[i]>dn:
			ll.append(i)
	print(path,len(ll))
	return ll


# 得到公共质合比
def get_vecs_key(paths,l,h,t):
	if t == 0:
		w = 'Train'
	else:
		w = 'Test'
	arr = []	
	for path in paths:
		v = stat_every(path,l,h,w)
		arr.append(v)
	vv = list(set(arr[0]+arr[1]))
	#print("The vecs-key is\n",vv)
	print(w+" Len(vecs-key) = ",len(vv))
	return vv


# 得到合并后的公共质合比
def get_fin_key():
	paths = [('Good',n1),('Bad',n1)]
	train_keys = get_vecs_key(paths,l,h,0)
	paths = [('Good',n2),('Bad',n2)]
	test_keys = get_vecs_key(paths,l,h,1)
	print('-----------------')
	fin_key = []
	for i in train_keys:
		if i in test_keys:
			fin_key.append(i)
	print('len_fin_vec:',len(fin_key))
	np.save('Exper_data/con_keys.npy',fin_key)


if __name__ == '__main__':
	get_fin_key()