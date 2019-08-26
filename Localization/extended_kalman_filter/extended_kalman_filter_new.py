"""

Extended kalman filter (EKF) localization sample

author: Atsushi Sakai (@Atsushi_twi)

"""

import math
import numpy as np

import sys
sys.path.append('..')
from localization import KalmanFilter
from simulator import Simulator
from history import History

show_animation = True

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]

#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)])**2
GPS_NOISE = np.diag([0.5, 0.5])**2

# Covariance for UKF simulation
Q = np.diag([
    0.1, # variance of location on x-axis
    0.1, # variance of location on y-axis
    np.deg2rad(1.0), # variance of theta
    # 1.0 # variance of velocity
    ])**2  # predict state covariance
R = np.diag([0.1, 0.1, 0.1])**2 

class MotionModel():
	
	def __init__(self):
		pass

	# g()
	def __call__(self, u, mu_past, delta=1, gamma=0):
		"""
			delta := scalar time step
			gamma := scalar to account for final rotation
			return state2 := (x_1, y_1, theta_1)
		"""
		print('Motion')
		print(u.shape)
		print(mu_past.shape)
		v, w = u
		x, y, theta = mu_past
		step = np.array([(-v/w) * np.sin(theta) + (v/w) * np.sin(theta + w * delta),
						(v/w) * np.cos(theta) - (v/w) * np.cos(theta + w * delta),
						(w + gamma) * delta])
		print(step.shape)

		mu_bar = mu_past + step
		print(mu_bar.shape)
		return mu_bar

	# G = dg
	def jacobian(self, u, mu_past):
		"""
		Jacobian of Motion Model

		motion model
		x_{t+1} = x_t+v*dt*cos(theta)
		y_{t+1} = y_t+v*dt*sin(theta)
		theta_{t+1} = theta_t+omega*dt
		v_{t+1} = v{t}
		so
		dx/dtheta = -v*dt*sin(theta)
		dx/dv = dt*cos(theta)
		dy/dtheta = v*dt*cos(theta)
		dy/dv = dt*sin(theta)
		"""
		theta = mu_past[2, 0]
		v, omega = u
		jacob = np.array([
			[1.0, 0.0, self.DT * (v/omega) * np.cos(theta) - self.DT * (v/omega) * np.cos(theta + omega * self.DT)],
			[0.0, 1.0, self.DT * (v/omega) * np.sin(theta) - self.DT * (v/omega) * np.sin(theta + omega * self.DT)],
			[0.0, 0.0, 1.0]])

		return jacob

class ObservationModel():
	
	def __init__(self):
		pass

	# h()
	def __call__(self, mu_bar):
		h = np.array([
			[1, 0, 0],
			[0, 1, 0]
		])
		print('h()')
		print(mu_bar.shape)
		print(h.shape)
		z = h @ mu_bar
		print('z: ', z.shape)
		return z

	# H = dh
	def jacobian(self, mu_bar):
		jacob = np.array([
			[1, 0, 0],
			[0, 1, 0]
		])

		return jacob

class EKF():
	
	def __init__(self, motion_model, observation_model, DT, Q, R):
		self.motion_model = motion_model
		self.observation_model = observation_model
		self.DT = DT
		self.Q = Q
		self.R = R

	# EKF (Table 3.3) & EKF_Localization_with_known_correspondences (Table 7.2)
	def __call__(self, mu_past, sigma_past, u, z):
		print(u_past.shape, sigma_past.shape, u.shape, z.shape)
		mu_bar, sigma_bar = self.prediction(mu_past, u)  # EKF lines 2-3
		mu, sigma = self.correction(mu_bar, sigma_bar, z)  # EKF lines 4-6
		return mu, sigma # EKF line 7

	def prediction(self, u, mu_past):
		mu_bar = self.g(u, mu_past) # EKF line 1
		G = self.jacob_g(u, mu_past)
		sigma_bar = G @ sigma_past @ G.T + self.R # EKF line 2
		return mu_bar, sigma_bar

	def g(self, u, mu_past):
		return self.motion_model(u, mu_past)

	def jacob_g(self, u, mu_past):
		return self.motion_model.jacobian(u, mu_past)

	def correction(self, mu_bar, sigma_bar, z):
		H = self.jacob_h(mu_bar)
		S = H @ sigma_bar @ H.T + self.Q
		K = sigma_bar @ H.T @ np.linalg.inv(S) # EKF line 4
		mu = mu_bar + K @ (z - self.h(mu_bar)) # EKF line 5
		I = np.eye(mu.shape[0])
		sigma = (I - K @ H) @ sigma_bar # EKF line 6
		return mu, sigma # EKF line 7

	def h(self, u, mu_past):
		return self.observation_model(u, mu_past)

	def jacob_h(self, u, mu_past):
		return self.observation_model.jacobian(u, mu_past)
	
def observation(observation_model, motion_model, pose, mu, u):
	print()
	print('OBSERVATION')
	pose = motion_model(u, pose) # F @ x + B @ u
	print('Pose: ', pose.shape)

	# add noise to gps x-y
	z = observation_model(pose) + GPS_NOISE @ np.random.randn(2, 1)
	print('z: ', z.shape)

	# add noise to input
	u_noisy = u + INPUT_NOISE @ np.random.randn(2, 1)

	mu = motion_model(mu, u_noisy)

	return xTrue, z, xd, ud

def main():
	print(__file__ + " start!!")

	time = 0.0

	# Initialize State Vector [x y theta]'
	mu = np.zeros((3, 1))
	pose = np.zeros((3, 1))
	sigma = np.eye(3)
	# print(mu.shape, pose.shape, sigma.shape)
	# print(Q.shape, R.shape)
	
	simulator = Simulator()
	motion_model = MotionModel()
	observation_model = ObservationModel()
	Localization = EKF(motion_model, observation_model, DT, Q, R)
	# history = History(mu, pose)


	while SIM_TIME >= time:
		time += DT

		u = simulator()
		# print(u.shape) # (2,1)
		pose, u_noisy, z = observation(observation_model, motion_model, pose, mu, u)
		mu, sigma = Localization(mu, sigma, u_noisy, z)

		# history.update(mu, xDR, pose, z)

		if show_animation:
			history.plot(mu, sigma)


if __name__ == '__main__':
    main()
