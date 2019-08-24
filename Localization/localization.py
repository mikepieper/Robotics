import numpy as np
import math

class BayesFilter():
	
	def __init__(self, DT):
		self.DT = DT

	def motion_model(self, x, u):
		F = np.array([[1.0, 0, 0, 0],
					[0, 1.0, 0, 0],
					[0, 0, 1.0, 0],
					[0, 0, 0, 0]])

		B = np.array([[self.DT * math.cos(x[2, 0]), 0],
					[self.DT * math.sin(x[2, 0]), 0],
					[0.0, self.DT],
					[1.0, 0.0]])

		x = F @ x + B @ u

		return x

	def observation(self):
		raise NotImplementedError
	
	def estimation(self):
		raise NotImplementedError

	
#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)])**2
GPS_NOISE = np.diag([0.5, 0.5])**2

class KalmanFilter(BayesFilter):
	
	def __init__(self, DT, Q, R):
		super().__init__(DT)
		self.Q = Q
		self.R = R


	def observation_model(self, x):
		H = np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0]
		])

		z = H @ x

		return z


	def observation(self, xTrue, xd, u):
		xTrue = self.motion_model(xTrue, u)

		# add noise to gps x-y
		z = self.observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

		# add noise to input
		ud = u + INPUT_NOISE @ np.random.randn(2, 1)

		xd = self.motion_model(xd, ud)

		return xTrue, z, xd, ud

	
	