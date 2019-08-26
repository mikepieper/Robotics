import numpy as np

class MotionModel():

    def __init__(self):
        pass

##################################

import numpy as np

class BayesFilter():
	
	def __init__(self, DT):
		self.DT = DT

	def motion_model(self, x, u):
		F = np.array([[1.0, 0, 0, 0],
					[0, 1.0, 0, 0],
					[0, 0, 1.0, 0],
					[0, 0, 0, 0]])

		B = np.array([[self.DT * np.cos(x[2, 0]), 0],
					[self.DT * np.sin(x[2, 0]), 0],
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

# Covariance for UKF simulation
Q = np.diag([
    0.1, # variance of location on x-axis
    0.1, # variance of location on y-axis
    np.deg2rad(1.0), # variance of yaw angle
    1.0 # variance of velocity
    ])**2  # predict state covariance

class KalmanFilter(BayesFilter):
	
	def __init__(self, DT, Q=Q):
		super().__init__(DT)
		self.Q = Q

	def compute_H(self, x, m):
		H = np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0]
		])
		return H

	def observation_model(self, x, m=None):
		H = self.compute_H(x, m)

		z = H @ x

		return z


	def observation(self, xTrue, xd, u):
		xTrue = self.motion_model(xTrue, u) # F @ x + B @ u

		# add noise to gps x-y
		z = self.observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1) # H @ x

		# add noise to input
		ud = u + INPUT_NOISE @ np.random.randn(2, 1)

		xd = self.motion_model(xd, ud) # F @ xd + B @ ud

		return xTrue, z, xd, ud
	
