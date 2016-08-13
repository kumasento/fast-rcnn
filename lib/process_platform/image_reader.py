
"""
Image reader that absorbs image from file path
"""

import os
import cv2

class ImageReader:
	def __init__(self):
		pass

	def read(self, image_path):
		if not os.path.isfile(image_path):
			print 'Cannot find image at %s' % image_path
			exit(1)
		img = cv2.imread(image_path)
		return img
