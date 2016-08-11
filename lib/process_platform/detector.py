"""
Detect class for each region
"""
import os
import caffe
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect

CLASSES = ('__background__',
			'aeroplane', 'bicycle', 'bird', 'boat',
			'bottle', 'bus', 'car', 'cat', 'chair',
			'cow', 'diningtable', 'dog', 'horse',
			'motorbike', 'person', 'pottedplant',
			'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16','vgg16_fast_rcnn_iter_40000.caffemodel'),
		'vgg_cnn_m_1024': ('VGG_CNN_M_1024', 'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
		'caffenet': ('CaffeNet', 'caffenet_fast_rcnn_iter_40000.caffemodel')}

class Detector:
	def __init__(self, net_name='caffenet'):

		# initialize caffe
		self._prototxt   = os.path.join(cfg.ROOT_DIR, 'models', NETS[net_name][0], 'test.prototxt')
		self._caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models', NETS[net_name][1])

		caffe.set_mode_gpu()
		caffe.set_device(0)
		self._net = caffe.Net(self._prototxt, self._caffemodel, caffe.TEST)
	
	def detect(self, img, candidates):
		"""
		img - the image for detection
		candidates - proposed regions' coordinates and shapes
		"""
		scores, boxes = im_detect(self._net, img, candidates)
		return scores, boxes