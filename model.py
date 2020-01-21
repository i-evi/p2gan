import tensorflow as tf
slim = tf.contrib.slim

def leaky_relu(x, lk = 0.2):
	return tf.maximum(x, x * lk)

def _fixed_padding(inputs, kernel_size, rate=1):
	kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
							kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
	pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
	pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
	pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
	padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
						[pad_beg[1], pad_end[1]], [0, 0]], mode='SYMMETRIC')
	return padded_inputs


DL1C = 256
DL2C = 512

discriminator_cfg_p9 = {
	'l_num': 2,
	'l0_c': 3,                           # 216
	'l1_c': DL1C,   'l1_k': 3, 'l1_s': 3, # 72
	'l2_c': DL2C,   'l2_k': 3, 'l2_s': 3, # 24
}

discriminator_cfg_p12 = {
	'l_num': 2,
	'l0_c': 3,                           # 240
	'l1_c': DL1C,   'l1_k': 4, 'l1_s': 4, # 60
	'l2_c': DL2C,   'l2_k': 3, 'l2_s': 3, # 20
}

discriminator_cfg_p15 = {
	'l_num': 2,
	'l0_c': 3,                           # 240
	'l1_c': DL1C,   'l1_k': 5, 'l1_s': 5, # 48
	'l2_c': DL2C,   'l2_k': 3, 'l2_s': 3, # 16
}

discriminator_cfg_p16 = {
	'l_num': 2,
	'l0_c': 3,                           # 256
	'l1_c': DL1C,   'l1_k': 4, 'l1_s': 4, # 64
	'l2_c': DL2C,   'l2_k': 4, 'l2_s': 4, # 16
}

supported_patch_size = {
	9 : discriminator_cfg_p9,
	12: discriminator_cfg_p12,
	15: discriminator_cfg_p15,
	16: discriminator_cfg_p16
}

batch_norm_decay=0.95
batch_norm_epsilon=0.001
batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS

def build_discriminator(inp, patch_size=9, is_training=True,
	name='discriminator', reuse=False):
	d_state = inp
	batch_norm_params = {
		'center': True,
		'scale': True,
		'decay': batch_norm_decay,
		'epsilon': batch_norm_epsilon,
		'updates_collections': batch_norm_updates_collections,
		'is_training': is_training
	}
	with slim.arg_scope([slim.batch_norm], **batch_norm_params):
		with tf.variable_scope(name, reuse=reuse):
			# conv s1
			cfg = supported_patch_size[patch_size]
			with slim.arg_scope([slim.conv2d],
				activation_fn=leaky_relu,
				normalizer_fn=slim.batch_norm,
				padding='VALID'):
				for l in range(1, cfg['l_num'] + 1):
					d_state = slim.conv2d(d_state,
						cfg['l%d_c'%l],
						[cfg['l%d_k'%l], cfg['l%d_k'%l]],
						stride=cfg['l%d_s'%l], scope='s1_%d'%l)
			d_state = slim.conv2d(d_state, 1, [1, 1], stride=1,
				activation_fn=tf.nn.sigmoid, scope='patch_mat')
	return d_state

g_encoder_cfg = {
	'l_num': 3,
	'l0_c': 32,   'l0_k': 3, 'l0_s': 2, # 1/2  --------> sc1
	'l1_c': 64,   'l1_k': 3, 'l1_s': 2, # 1/4  --------> sc2
	'l2_c': 128,  'l2_k': 3, 'l2_s': 2  # 1/8  --------> sc3 -L
}

g_residual_cfg = {
	'l_num': 1,
	'c': 128, 'k': 3
}

g_decoder_cfg = {
	'l_num': 3,
	'l0_c': 64,  'l0_k': 3, 'l0_s': 2, # --- x2
	'l1_c': 32,  'l1_k': 3, 'l1_s': 2, # --- x4
	'l2_c': 16,  'l2_k': 3, 'l2_s': 2, # --- x8
}

g_skip_conn_cfg = {
	'l_num' : 2
}

def build_generator(inp, name='generator', reuse=False):
	g_state = inp   # No prep
	with tf.variable_scope(name, reuse=reuse):
		cfg = g_encoder_cfg
		skip_conn = []
		with slim.arg_scope([slim.conv2d, slim.separable_conv2d], activation_fn=tf.nn.relu,
					normalizer_fn=slim.instance_norm, padding='VALID'):
			for index in range(cfg['l_num']):
				g_state = _fixed_padding(g_state, [cfg['l%d_k'%index]])
				g_state = slim.separable_conv2d(g_state, None, [cfg['l%d_k'%index], cfg['l%d_k'%index]],
					depth_multiplier=1, stride=cfg['l%d_s'%index], scope='enc_%d_dw'%index)
				g_state = slim.conv2d(g_state, cfg['l%d_c'%index], [1, 1], stride=1, scope='enc_%d_pw'%index)
				skip_conn.append(g_state)

		cfg = g_residual_cfg
		with slim.arg_scope([slim.conv2d, slim.separable_conv2d], activation_fn=tf.nn.relu,
					normalizer_fn=slim.instance_norm, padding='VALID'):
			for index in range(cfg['l_num']):
				res_g = g_state
				g_state = _fixed_padding(g_state, [cfg['k']])
				g_state = slim.separable_conv2d(g_state, None, [cfg['k'], cfg['k']],
					depth_multiplier=1, stride=1, scope='res_%d_dw'%index)
				g_state = slim.conv2d(g_state, cfg['c'], [1, 1], stride=1,
					activation_fn=None, scope='res_%d_pw'%index)
				g_state = tf.nn.relu(g_state + res_g)

		cfg = g_decoder_cfg
		with slim.arg_scope([slim.conv2d, slim.separable_conv2d], activation_fn=None,
					normalizer_fn=slim.instance_norm, padding='VALID'):
			for index in range(cfg['l_num']):
				g_state = tf.image.resize_images(g_state,
					(g_state.shape[1]*2, g_state.shape[2]*2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
				g_state = _fixed_padding(g_state, [cfg['l%d_k'%index]])
				g_state = slim.separable_conv2d(g_state, None,
					[cfg['l%d_k'%index], cfg['l%d_k'%index]], depth_multiplier=1, stride=1, scope='dec_%d_dw'%index)
				g_state = slim.conv2d(g_state, cfg['l%d_c'%index], [1, 1], stride=1, scope='dec_%d_pw'%index)
				# sc = slim.conv2d(skip_conn[?], 128, [1, 1], stride=1, scope='sc')
				if index < g_skip_conn_cfg['l_num']:
					g_state = tf.nn.relu(g_state + skip_conn[g_skip_conn_cfg['l_num'] - index - 1])
				else:
					g_state = tf.nn.relu(g_state)
		g_state = _fixed_padding(g_state, [3])
		g_state = slim.conv2d(g_state, 3, [3, 3], stride=1, padding='VALID',
			activation_fn=tf.nn.tanh, normalizer_fn=None, scope='output')
	return g_state
