import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_hotmap():
	hot_path = 'hots.npy'
	hot = np.load(hot_path) 
	'''
	res = []
	for i in hot:
		if len(list(np.nonzero(i)[0])) < 20:
			continue
		else:
			res.append(i)
	'''
	hots = []
	for i in hot:
		tmp = []
		for j in i:
			for k in range(4):
				tmp.append(j)
		hots.append(tmp)
	hotsss = hots
	for i in range(31):
		hotsss = np.concatenate((hotsss,hots),axis = 1)
	'''
	hotps = []
	for j in range(len(hotsss)):
		hotp = []
		for i in range(len(hotsss[j])):
			if i % 2 == 1:
				hotp.append(hotsss[0][i]+hotsss[0][i-1])
		hotps.append(hotp)


	hotps = np.array(hotps)
	'''
	heatmap = np.uint8(255 * hotsss)
	#heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	plt.axis('off')
	plt.imshow(heatmap)
	plt.savefig('hotsss.png')
	'''
	plt.figure()
	plt.xticks([])
	plt.yticks([])
	plt.xlabel("samples")
	plt.ylabel("m/z")
	'''
	hots = np.array(hots).T
	hot1 = hots[:,:111]
	hot2 = hots[:,111:]
	heatmap = np.uint8(255 * hots)
	heatmap1 = np.uint8(255 * hot1)
	heatmap2 = np.uint8(255 * hot2)
	plt.imshow(heatmap)
	plt.savefig('hots.png')
	plt.imshow(heatmap1)
	plt.savefig('hot1.png')
	plt.imshow(heatmap2)
	plt.savefig('hot2.png')
	color = []
	for i in range(256):
		color.append([i]*100)
	plt.imshow(color)
	plt.savefig('color.png')


if __name__ == '__main__':
	get_hotmap()
