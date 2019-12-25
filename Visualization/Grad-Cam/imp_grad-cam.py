# -*- coding: utf-8 -*-
import tensorflow
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras import models
from keras import backend as K
import matplotlib.pyplot as plt


def resize(x):
	return np.reshape(x,(1,256,16))


def get_hots_npy():
	model = load_model('model.h5')	#加载模型
	model.summary()

	np_name = np.load("datas/X.npy")

	hots = []
	for i in np_name:
		np_pred = resize(i)
		preds = model.predict(np_pred)	#预测值
		p = preds[0]
		if p[0] > p[1]:
			pr = 0
		else:
			pr = 1
		case_output = model.output[:, pr]	#数值为类别号
		last_conv_layer = model.get_layer('conv1d_3')	#最后一层卷积层的名称
		grads = K.gradients(case_output, last_conv_layer.output)[0]
		pooled_grads = K.mean(grads, axis=(0,1,2))
		iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
		pooled_grads_value, conv_layer_output_value = iterate([np_pred])

		for j in range(256):	#最后一层卷积层的特征图数量
			conv_layer_output_value[:, j] *= pooled_grads_value

		heatmap = np.mean(conv_layer_output_value, axis=-1)
		heatmap = np.maximum(heatmap, 0)
		heatmap /= np.max(heatmap)#ok 热图
		hots.append(heatmap)
		np.save('hots.npy',hots)


if __name__ == '__main__':
	get_hots_npy()
