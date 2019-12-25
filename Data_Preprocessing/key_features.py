import numpy as np
import os
import sys
import threading


r = sys.argv[1] #
l = int(sys.argv[2]) #15
h = int(sys.argv[3]) #95


# 组合时间、强度和质荷比的信息成特征
# 每个质荷比时间二等分、强度取强度值的和
def get_vecs_1(path,lists,low,high):
	files = os.listdir(path)
	for l in files:
		res = []
		arr_0_60 = []
		arr_60_120 = []
		f = np.load(path+'/'+l)
		x0 = f[:,0]
		x1 = f[:,1]
		x2 = f[:,2]
		me1,me2 = np.percentile(x1, [low, high])
		for i in range(len(x1)):
			if x1[i] >= me1 and x1[i] <= me2 and x0[i] <= 40:
				arr_0_60.append(i)
			if x1[i] >= me1 and x1[i] <= me2 and x0[i] > 40:
				arr_60_120.append(i)
		ff1 = f[arr_0_60] # 先按时间分段
		ff2 = f[arr_60_120]

		for j in lists:
			tmp = np.argwhere(ff1[:,2]==j)
			inds = []
			for i in tmp:
				inds.append(i[0])
			ss1 = ff1[inds][:,1]
			if ss1.tolist() == []:
				ss1 = [0]
			res.append(np.sum(ss1))
			tmp = np.argwhere(ff2[:,2]==j)
			inds = []
			for i in tmp:
				inds.append(i[0])
			ss2 = ff2[inds][:,1]
			if ss2.tolist() == []:
				ss2 = [0]
			res.append(np.mean(ss2))
		p = path.split('/')[0]+"/Fea-"+path.split('/')[1]+"/"+l
		np.save(p,res)
		print(p)


# 不考虑时间、强度取强度值的和的平均值和标准差
def get_vecs_2(path,lists,low,high):
	files = os.listdir(path)
	for l in files:
		res = []
		arr = []
		f = np.load(path+'/'+l)
		x0 = f[:,0]
		x1 = f[:,1]
		x2 = f[:,2]
		me1,me2 = np.percentile(x1, [low, high])
		for i in range(len(x1)):
			if x1[i] >= me1 and x1[i] <= me2:
				arr.append(i)
		ff = f[arr]
		for j in lists:
			tmp = np.argwhere(ff[:,2]==j)
			inds = []
			for i in tmp:
				inds.append(i[0])
			ss = ff[inds][:,1]
			if ss.tolist() == []:
				ss = [0]
			res.append(np.sum(ss))
		p = path.split('/')[0]+"/Fea-"+path.split('/')[1]+"/"+l
		np.save(p,res)
		print(p)


dic = {'1':get_vecs_1, '2':get_vecs_2}


# 得到全部特征向量
def get_features(key,l,h,get_vecs):
	paths = ['Train/Good','Train/Bad', 'Test/Good','Test/Bad']
	for path in paths:
		get_vecs(path,key,l,h)


if __name__ == '__main__':
	key = np.load("Exper_data/con_keys.npy")
	get_features(key,l,h,dic[r])


'''
# 得到全部特征向量
def get_features(key,l,h,get_vecs):
	paths = ['Train/Good','Train/Bad', 'Test/Good','Test/Bad']
	threads = []
	for path in paths:
		t = threading.Thread(target=get_vecs,args=(path,key,l,h,))
		threads.append(t)
	for t in threads:
		t.setDaemon(True)
		t.start()
	t.join()


if __name__ == '__main__':
	key = np.load("Exper_data/con_keys.npy")
	get_features(key,l,h,dic[r])
'''