import cv2
import json
import numpy as np
import os
import random

def imbatch_read(path, file_list, shape):
	if len(file_list) <= 0:
		return None
	sls = []
	batch = None
	for i, item in enumerate(file_list):
		img = cv2.imread(path +  item)
		sls.append((img.shape[1], img.shape[0]))
		img = cv2.resize(img, shape)
		img = (img / 255.0 - 0.5) * 2.0
		img = np.expand_dims(img, axis=0)
		if not isinstance(batch, np.ndarray):
			batch = img
		else:
			batch = np.vstack((batch, img))
	return batch, sls

def img_write(img, path, shape):
	if len(img.shape) != 3 or img.shape[2] != 3:
		return False # Only support RGB img
	img = (((img + 1) / 2) * 255)#.astype(np.uint8)
	img = cv2.resize(img, shape)
	cv2.imwrite(path, img)
	return True

def background(shape, ch=3):
	return np.zeros(shape[0] * shape[1] * ch).\
		astype(np.uint8).reshape((shape[0], shape[1], ch))

def crop(img, x, y, xl, yl):
	if (x + xl) > img.shape[1] or (y + yl) > img.shape[0]:
		return None
	return img[y:y+yl, x:x+xl,:]

def merge(bg, img, x, y):
	bg[y:y+img.shape[0] if y+img.shape[0] < bg.shape[0] else bg.shape[0],
		x:x+img.shape[1] if x+img.shape[1] < bg.shape[1] else bg.shape[1],:] =\
			img[:img.shape[0] if y+img.shape[0] < bg.shape[0] else bg.shape[0] - y,
		:img.shape[1] if x+img.shape[1] < bg.shape[1] else bg.shape[1] - x,:]

def random_crop(img, xl, yl):
	y = int((img.shape[0] - yl) * random.random())
	x = int((img.shape[1] - xl) * random.random())
	return crop(img, x, y, xl, yl)

def patch_fill(bg, img, offset_x, offset_y, gap, s):
	if bg.shape[0] != bg.shape[1]:
		exit('Error filling patches, background or image must be square')
	# if s % 2 != 1:
	# 	exit('Patch size must be odd, not %d'%s)
	jump = s + gap
	cnt = int(bg.shape[0] / (jump))
	for j in range(cnt):
		for i in range(cnt):
			merge(bg, random_crop(img, s, s),
				i * jump + offset_x, j * jump + offset_y)
	return bg

def patch_fill_multi(bg, imgls, offset_x, offset_y, gap, s):
	if bg.shape[0] != bg.shape[1]:
		exit('Error filling patches, background or image must be square')
	# if s % 2 != 1:
	# 	exit('Patch size must be odd, not %d'%s)
	jump = s + gap
	cnt = int(bg.shape[0] / (jump))
	for j in range(cnt):
		for i in range(cnt):
			merge(bg, 
				random_crop(imgls[int(random.random() * len(imgls))], s, s),
				i * jump + offset_x, j * jump + offset_y)

def load_cfg(filename):
	with open(filename, 'r') as fp:
		return json.loads(fp.read())

def pickup_list(ls, cnt, begin):
	ls_len = len(ls)
	if begin >= ls_len:
		return None
	if (begin + cnt) >= ls_len:
		begin = ls_len - cnt
	return ls[begin:begin+cnt]

def make_input_batch(img, bs, h, w, ps):
	(_, _, c) = img.shape
	bat = np.zeros(bs * h * w * c).astype(np.float32).reshape((bs, h, w, c))
	for i in range(bs):
		patch_fill(bat[i], img, 0, 0, 0, ps)
	return bat


def make_input_batch_multi(imgls, bs, h, w, ps):
	(_, _, c) = imgls[0].shape
	bat = np.zeros(bs * h * w * c).astype(np.float32).reshape((bs, h, w, c))
	for i in range(bs):
		patch_fill_multi(bat[i], imgls, 0, 0, 0, ps)
	return bat

def ls_files_to_json(path, jname=None, reverse=False, ext=[]):
	ls = []
	for x in os.listdir(path):
		if os.path.isfile(os.path.join(path, x)):
			if len(ext) == 0:
				ls.append(x)
			else:
				if x.split('.')[-1] in ext:
					ls.append(x)
	if reverse == None:
		pass
	elif reverse == False:
		ls.sort(reverse=False)
	elif reverse == True:
		ls.sort(reverse=True)
	if isinstance(jname, str):
		with open(jname, 'w+') as fp:
			fp.write(json.dumps(ls))
	return ls

def open_img(img_path):
	img = cv2.imread(img_path)
	img = (img / 255.0 - 0.5) * 2.0
	return img

def load_bat_img(img_path, shape, prep=True, singleCh=False, gray=False, remove_pad=False):
	img = cv2.imread(img_path)
	if gray:
		img= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).reshape((img.shape[0], img.shape[1], 1))
		timg = np.copy(img)
		img = np.concatenate((img, timg), axis=2)
		img = np.concatenate((img, timg), axis=2)
	if remove_pad:
		y = img.shape[0]
		x = img.shape[1]
		img = img[int(y/8):y - int(y/8), int(x/8):x-int(x/8),:]
	if isinstance(shape, tuple):
		img = cv2.resize(img, shape)
	img = np.expand_dims(img, axis=0)
	if prep:
		img = (img / 255.0 - 0.5) * 2.0
	if singleCh:
		img = np.mean(img, axis=3)
		img = np.expand_dims(img, axis=3)
	return img

def random_file_list(path, batch_size, rand=True):
	images_path =[n for n in os.listdir(path)]
	if (batch_size > len(images_path) or batch_size <= 0):
		print('N/A')
		return None
	if rand:
		random.shuffle(images_path)
	return images_path[0:batch_size]

def images_batch(folder, file_list, shape, prep=True, singleCh=False, gray=False, remove_pad=False):
	if len(file_list) <= 0:
		return None
	batch = load_bat_img(folder+'/'+file_list[0], prep=prep, shape=shape, singleCh=False, gray=gray, remove_pad=remove_pad)
	for i in range(len(file_list) - 1):
		batch = np.vstack((batch, load_bat_img(folder+'/'+file_list[i + 1], prep=prep, shape=shape, singleCh=False, gray=gray, remove_pad=remove_pad)))
	return batch

def save_as_rgb_img(img, path):
	if len(img.shape) != 3 or img.shape[2] != 3:
		return False # Only support RGB img
	img = (((img + 1) / 2) * 255)
	cv2.imwrite(path, img)
	return True

def save_batch_as_rgb_img(bat, path, prefix=None):
	code = 0
	for item in bat:
		if prefix == None:
			save_as_rgb_img(item, path+'/%d.jpg'%code)
		else:
			save_as_rgb_img(item, path+'/%s%d.jpg'%(prefix, code))
		code = code + 1

def random_sub_set(set, n):
	if n <= 1:
		return set
	set_index = np.arange(1, set.shape[0] + 1)
	subset_index = np.random.choice(set_index, n)
	batch_data = set10[set10_index[0]]
	batch_data = np.expand_dims(batch_data, axis=0)
	for item in subset_index[1:]:
		batch_data = np.vstack((batch_data, np.expand_dims(set10[item], axis=0)))
	return batch_data

def sub_set(inset, index_tab):
	if len(index_tab) <= 1:
		return inset
	batch_data = inset[index_tab[0]]
	batch_data = np.expand_dims(batch_data, axis=0)
	for item in index_tab[1:]:
		batch_data = np.vstack((batch_data, np.expand_dims(inset[item], axis=0)))
	return batch_data

def silent_mkdir(path):
	try:
		os.makedirs(path)
	except:
		pass
