import cv2
import numpy as np
import tensorflow as tf
import util
import time
import model
import os
from argparse import ArgumentParser

imbatch_read     = util.imbatch_read
img_write        = util.img_write
pickup_list      = util.pickup_list
ls_files_to_json = util.ls_files_to_json

build_generator     = model.build_generator

def build_parser():
	parser = ArgumentParser()
	parser.add_argument('--model', type=str,
					dest='model',
					help='dir to load model',
					required=True)
	parser.add_argument('--inp', type=str,
					dest='inp_path',
					help='input content images',
					required=True)
	parser.add_argument('--oup', type=str,
					dest='out_path',
					help='output stylized images',
					required=True)
	parser.add_argument('--bs', type=int,
					dest='batch_size',
					default=1,
					help='batch size',
					required=False)
	parser.add_argument('--size', type=int,
					dest='net_size',
					default=256,
					help='network feed size',
					required=False)
	parser.add_argument('--cpu', type=str,
					dest='use_cpu',
					default='false',
					help='processing by cpu',
					required=False)
	parser.add_argument('--noise', type=float,
					dest='noise',
					default=0.0,
					help='noise',
					required=False)
	return parser

args = build_parser().parse_args()

MODEL_SAVE_PATH = args.model

BATCH_SIZE  = args.batch_size
FEED_SIZE   = args.net_size
IMGSRC_PATH = args.inp_path
NOISE_RATE  = args.noise

if (not os.path.isdir(args.out_path)):
	os.makedirs(args.out_path)

DEVICE = ''
if args.use_cpu.upper()=='TRUE':
	DEVICE = '/cpu:0'

input_ls = ls_files_to_json(IMGSRC_PATH, ext=['png', 'bmp', 'jpg', 'jpeg'])
CNT = len(input_ls)

gpu_options = tf.GPUOptions(allow_growth=True)

with tf.device(DEVICE), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	input_r = tf.placeholder(tf.float32, shape=[BATCH_SIZE, FEED_SIZE, FEED_SIZE, 3], name='inpr')
	g_state = build_generator(input_r, name='generator')
	g_var_ls = tf.trainable_variables(scope='generator')
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(g_var_ls)
	chkpt_fname = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
	saver.restore(sess, chkpt_fname)
	# Warm up network and test...
	noise = np.random.normal(0, 1, size=(BATCH_SIZE, FEED_SIZE, FEED_SIZE, 3)).astype(np.float32)
	sess.run(g_state, feed_dict={input_r: noise})
	# Begin...
	total = int(CNT / BATCH_SIZE) + (1 if (CNT % BATCH_SIZE) != 0 else 0)
	for offset in range(total):
		sub_ls  = pickup_list(input_ls, BATCH_SIZE, offset * BATCH_SIZE)
		sub_img, sls = imbatch_read(IMGSRC_PATH, sub_ls, (FEED_SIZE, FEED_SIZE))
		if NOISE_RATE != 0:
			noise = np.random.normal(0, 1,
				size=(BATCH_SIZE, FEED_SIZE, FEED_SIZE, 3)).astype(np.float32)
			sub_img = sub_img + noise * NOISE_RATE 
		time_start = time.time()
		render_batch = sess.run(g_state, feed_dict={input_r: sub_img})
		print('Processing: %d/%d, network dataflow time: %f'%(offset + 1, total, time.time()-time_start))
		for i in range(len(sls)):
			img_write(render_batch[i],
					args.out_path+'/%d_%d.jpg'%(offset, i),
				sls[i])# (FEED_SIZE, FEED_SIZE)) #