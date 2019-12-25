import os
import re
import threading
import numpy as np
import sys


train_path = 'Raw_Data/IPX0000937001XIC1_3_10/'
test_path = 'Raw_Data/IPX0001046000part/'
nums = int(sys.argv[1])


# 判断是癌还是癌旁
def split_(ii):
	for j in ii[::-1]:
		if j not in ['A','T','P','B']:
			continue
		else:
			if j in ['A','T']:
				return 1
			else:
				return 0


# 得到训练样本的癌和癌旁的文件列表
def get_train_lis():
	train_raw_data = os.listdir(train_path)
	pb_train = []
	at_train = []
	for i in train_raw_data:
		if split_(i) == 1:
			at_train.append(i)
		if split_(i) == 0:
			pb_train.append(i)
	return at_train,pb_train


# 得到训练样本的癌和癌旁的文件列表
def get_test_lis():
	at_train = os.listdir(test_path+'/T')
	pb_train = os.listdir(test_path+'/P')
	return at_train,pb_train


# 得到训练数据病例的字典key
def get_train_dkey(ff):
	for i,j in enumerate(ff[::-1]):
		if j not in ['A','T','P','B']:
			continue
		else:
			return ff[-i-8:-i]	


# 得到测试数据病例的字典key
def get_test_dkey(ff):
	return ff[:15]


# 得到数据病例的字典key
def get_dkey(t, i):
	if t == 0:
		return get_train_dkey(i)
	else:
		return get_test_dkey(i)


# 提取文件对应列数据
def load_data(file_path, num):
	ll = np.array([3,9,12])
	bars = []
	f = open(file_path,'r')
	data = f.readlines()
	for ii in data[1:-1]:
		cc = ii.split('\n')[0]
		cc = cc.split('\t')
		cc = (np.array(list(map(float,cc))))[ll]
		cc[2] = float(re.findall(r"\d{1,}?\.\d{"+str(num)+"}", str(cc[2])+'0'*num)[0])
		bars.append(cc)
	#bars = sorted(bars, key=lambda arr: arr[1])
	return bars


# 得到病例-数据文件的字典
def get_at_dic(at_train, pb_train, t):
	pb_dic = {}
	for i in pb_train:
		ind = get_dkey(t,i)
		if ind not in pb_dic.keys():
			pb_dic[ind] = [i]
		else:
			pb_dic[ind].append(i)
	at_dic = {}	
	for i in at_train:
		ind = get_dkey(t,i)
		if ind not in at_dic.keys():
			at_dic[ind] = [i]
		else:
			at_dic[ind].append(i)
	return pb_dic, at_dic


# 根据上一步得到的关系字典将每一个病例的数据文件提取后合并成一个文件保存至本地
def save_data(pb_dic, at_dic, t, num):
	if t == 0:
		p = 'Train/'
		f1 = train_path
		f2 = train_path
	else:
		p = 'Test/'
		f1 = test_path+'/P/'
		f2 = test_path+'/T/'
	for i in pb_dic.items():
		n = 0
		for j in i[1]:
			print(j)
			gg = load_data(f1+j, num)
			if n == 0:
				rr = gg
				n = 1
			else:			
				rr = np.concatenate((rr,gg),axis = 0)
		np.save(p+'Good/'+str(i[0])+'.npy',rr)
	for i in at_dic.items():
		n = 0
		for j in i[1]:
			print(j)
			gg = load_data(f2+j, num)
			if n == 0:
				rr = gg
				n = 1
			else:
				rr = np.concatenate((rr,gg),axis = 0)
		np.save(p+'Bad/'+str(i[0])+'.npy',rr)



if __name__ == '__main__':
	at_train, pb_train = get_train_lis()
	pb_dic, at_dic = get_at_dic(at_train, pb_train, 0)
	save_data(pb_dic, at_dic, 0, 3)

	at_test, pb_test = get_test_lis()
	pb_dic, at_dic = get_at_dic(at_test, pb_test, 1)
	save_data(pb_dic, at_dic, 1, 3)


'''
def pre_train(s):
	at_train, pb_train = get_train_lis()
	pb_dic, at_dic = get_at_dic(at_train, pb_train, 0)
	save_data(pb_dic, at_dic, 0, nums)
	print(s)


def pre_test(s):
	at_test, pb_test = get_test_lis()
	pb_dic, at_dic = get_at_dic(at_test, pb_test, 1)
	save_data(pb_dic, at_dic, 1, nums)
	print(s)


threads = []
t1 = threading.Thread(target=pre_train,args=(u'Train',))
threads.append(t1)
t2 = threading.Thread(target=pre_test,args=(u'Test',))
threads.append(t2)

if __name__ == '__main__':
    for t in threads:
        t.setDaemon(True)
        t.start()
    t.join()
'''