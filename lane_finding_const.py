# constants for advanced_lane_lines.py
import numpy as np

# Perspective Transform
# source points
src = np.float32([[568, 470], [718, 470], [1110, 720], [210, 720]])
# destination points
dst = np.float32([[300, 0], [980, 0], [980, 720], [300, 720]])

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720 # meters per pixel in y dimension
xm_per_pix = 3.7 / 700 # meters per pixel in x dimension

# Low Pass FIR depth
N_LP = 5

# Max no. of continus bad frames before full frame lane searching reset
N_MISS = 5
