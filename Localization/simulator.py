import numpy as np

class ControlModel():
	
	def __init__(self):
		pass

	def __call__(self):
		v = 1.0  # [m/s]
		omega = 0.1  # [rad/s], yawrate
		u = np.array([[v, omega]]).T
		return u

