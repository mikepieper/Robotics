"""

Extended kalman filter (EKF) localization sample

author: Atsushi Sakai (@Atsushi_twi)

"""

import math
import numpy as np

from history import History

import sys
sys.path.append('..')
from simulator import ControlModel


show_animation = True

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]

#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)])**2
GPS_NOISE = np.diag([0.5, 0.5])**2

# Covariance for EKF simulation
R = np.diag([
    0.1, # variance of location on x-axis
    0.1, # variance of location on y-axis
    np.deg2rad(1.0), # variance of yaw angle
    1.0 # variance of velocity
    ])**2  # predict state covariance
Q = np.diag([1.0, 1.0])**2  # Observation x,y position covariance

class MotionModel():
	
	def __init__(self):
		pass

	# g()
	def __call__(self, u, mu_past):
		F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

		theta = mu_past[2, 0]
		B = np.array([[DT * np.cos(theta), 0],
					[DT * np.sin(theta), 0],
					[0.0, DT],
					[1.0, 0.0]])

		mu_bar = F @ mu_past + B @ u
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

		v, omega = u
		theta = mu_past[2,0]
		jacob = np.array([
			[1.0, 0.0, -DT * v * np.sin(theta), DT * np.cos(theta)],
			[0.0, 1.0, DT * v * np.cos(theta), DT * np.sin(theta)],
			[0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 1.0]])

		return jacob

class ObservationModel():
	
	def __init__(self):
		pass

	# h()
	def __call__(self, mu_bar):
		h =  np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0]
		])

		z = h @ mu_bar
		return z

	# H = dh
	def jacobian(self, mu_bar):
		jacob = np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0]
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
		mu_bar, sigma_bar = self.prediction(u, mu_past, sigma_past)  # EKF lines 2-3
		mu, sigma = self.correction(mu_bar, sigma_bar, z)  # EKF lines 4-6
		return mu, sigma # EKF line 7

	def prediction(self, u, mu_past, sigma_past):
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

	def h(self, mu_past):
		return self.observation_model(mu_past)

	def jacob_h(self, mu_past):
		return self.observation_model.jacobian(mu_past)
	
def observation(observation_model, motion_model, pose, mu, u):
	pose = motion_model(u, pose) # F @ x + B @ u

	# add noise to gps x-y
	z = observation_model(pose) + GPS_NOISE @ np.random.randn(2, 1)

	# add noise to input
	u_noisy = u + INPUT_NOISE @ np.random.randn(2, 1)

	mu = motion_model(u_noisy, mu)

	return pose, u_noisy, z, mu

def main():
	print(__file__ + " start!!")

	time = 0.0

	# Initialize State Vector [x y theta v]'
	mu = np.zeros((4, 1))
	pose = np.zeros((4, 1))
	sigma = np.eye(4)

	xDR = np.zeros((4, 1))  # Dead reckoning, mu

	control_model = ControlModel()
	motion_model = MotionModel()
	observation_model = ObservationModel()
	Localization = EKF(motion_model, observation_model, DT, Q, R)
	history = History(mu, pose)


	while SIM_TIME >= time:
		time += DT

		u = control_model()
		pose, u_noisy, z, mu = observation(observation_model, motion_model, pose, mu, u)
		xDr = np.copy(mu) # Dead reckoning
		mu, sigma = Localization(mu, sigma, u_noisy, z)

		history.update(mu, xDr, pose, z)

		if show_animation:
			history.plot(mu, sigma)


if __name__ == '__main__':
    main()
