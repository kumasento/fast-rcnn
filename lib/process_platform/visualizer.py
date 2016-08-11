
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Visualizer:
	def __init__(self):
		pass

	def visualize_regions(self, img, class_name, dets):
		"""
		img - the image
		dets - result from labelizer
		"""
		# draw the image for all regions found
		fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
		ax.imshow(img)
		for det in dets:
			print det
			x, y, h, w, score = det
			ax.add_patch(patches.Rectangle((x,y), h, w, fill=False, edgecolor='red'))
			ax.text(x, y-2, '{:s} {:.3f}'.format(class_name, score), 
							bbox=dict(facecolor='blue', alpha=0.5),  
							fontsize=14, color='white')
		plt.show()