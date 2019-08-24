import numpy as np

class Simulator():
	
	def __init__(self):
		pass

	def calc_input(self):
		v = 1.0  # [m/s]
		yawrate = 0.1  # [rad/s]
		u = np.array([[v, yawrate]]).T
		return u

