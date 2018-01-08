import numpy as np
from matplotlib import pyplot as plt

class HeatmapHistory():
    def __init__(self, heatmap_dim=(720, 1280), capacity=10, threshold=5):
        self.capacity = capacity
        self.heatmaps = []
        self.dimensions = heatmap_dim
        self.curr_count = 0
        self.threshold = threshold

    # Using Udacity's suggested heatmap functions or modifying them to taste
    def add(self, bbox_list):
        # Iterate through list of bboxes
        heatmap = np.zeros(self.dimensions, dtype=np.uint8)
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Return updated heatmap
        if self.capacity > self.curr_count:
            self.heatmaps.append(heatmap)
            self.curr_count += 1
        else:
            self.heatmaps.pop(0)
            self.heatmaps.append(heatmap)

    def get_heatmap(self):
        heatmap = np.sum(self.heatmaps, axis=0)
        #print(heatmap.shape)
        # Zero out pixels below the threshold
        heatmap[heatmap <= self.threshold] = 0
        heatmap = np.clip(heatmap, 0, 255)
        # Return thresholded map
        return heatmap

    def visualize_heatmap(self):
        plt.imshow(self.get_heatmap(), cmap='hot')
        plt.title("Thresholded heatmap")
        plt.show()
