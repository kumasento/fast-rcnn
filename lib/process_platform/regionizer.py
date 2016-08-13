"""
Use selective search to propose regions in the image
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import selectivesearch as ss

class Regionizer:
	def __init__(self, scale=500, sigma=0.9, min_size=10, min_patch=100, 
		min_distort=1.2):
		# configure the parameters for selective search
		self._scale = scale
		self._sigma = sigma
		self._min_size = min_size
		self._min_patch = min_patch
		self._min_distort = min_distort

	def regionize(self, img):
		img_lbl, regions = ss.selective_search(img, scale=self._scale, sigma=self._sigma, 
			min_size=self._min_size)
		candidates = set()
		for r in regions:
			if r['rect'] in candidates:
				continue
			if r['size'] < self._min_patch: # min patch size threshold
				continue
			x, y, h, w = r['rect']
			if w/h > 1.2 or h/w > 1.2: # distorted
				continue
			candidates.add(r['rect'])
		return np.array(list(candidates))

	def show_regions(self, img):
		candidates = self.regionize(img)
		fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
		ax.imshow(img)
		for x, y, h, w in candidates:
			ax.add_patch(patches.Rectangle((x,y), h, w, fill=False, edgecolor='red'))
		plt.show()
		return candidates