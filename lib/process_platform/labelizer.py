import numpy as np
from utils.cython_nms import nms
from process_platform.detector import CLASSES

class Labelizer:
	def __init__(self, threshold=0.8, nms_threshold=0.3):
		self._threshold = threshold
		self._nms_threshold = nms_threshold

	def labelize(self, scores, boxes, classes=[]):
		result = {}
		for cls in classes:
			cls_ind = CLASSES.index(cls)
			# get boxes and scores for this current class
			cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
			cls_scores = scores[:, cls_ind]
			print 'Max score for class %s: %lf' % (cls, np.max(cls_scores))
			# for those related regions, find out whether to keep or discard by using score.
			# keep is the index mask
			keep = np.where(cls_scores >= self._threshold)[0]
			cls_boxes = cls_boxes[keep, :]
			cls_scores = cls_scores[keep]

			# compress these two arrays together
			dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
			keep = nms(dets, self._nms_threshold)
			dets = dets[keep, :]

			result[cls] = dets
		return result