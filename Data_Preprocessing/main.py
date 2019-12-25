import os

# 质荷比保留小数点位数 | dot (0,1,2,3,4)
# 公共质荷比下限 | pub_dn pub_dn
# 统计公共质荷比的极值过滤参数 | stat_up stat_dn（%）
# 按照公共质荷比提取特征向量时的特征提取方案 | plan
# 强度计算的极值过滤参数 | I_up I_dn（%）
# 预分类块大小 块移动步长 选择的最优块的比例 | piece step opt（%）

args = {'dot':3,
		'tr_dn':0.2,
		'ts_dn':0.2,
		'stat_down':15,
		'stat_up':95,
		'plan':1,
		'I_down':15,
		'I_up':95,
		'piece':100,
		'step':10,
		'opt':10}


if __name__ == '__main__':
	os.system('python3 extract_raw_data.py 3')
	os.system('python3 stat_con_keys.py 22 16 15 95')
	os.system('python3 key_features.py 1 15 95')
	os.system('python3 pre_classify.py 100 10 10')
	os.system('python3 ml_model.py '+str(args))
