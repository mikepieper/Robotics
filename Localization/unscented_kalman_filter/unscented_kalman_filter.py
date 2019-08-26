"""

Unscented kalman filter (UKF) localization sample

author: Atsushi Sakai (@Atsushi_twi)

"""

import numpy as np
import scipy.linalg
import math
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from localization import KalmanFilter
from simulator import Simulator
from history import History

show_animation = True

simulator = Simulator()


R = np.diag([1.0, 1.0])**2  # Observation x,y position covariance

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]



show_animation = True


# DEFAULT PARAMS
R = np.diag([1.0, 1.0])**2  # Observation x,y position covariance

#  UKF Parameter
ALPHA = 0.001
BETA = 2
KAPPA = 0

class UnscentedKF(KalmanFilter):
	
	def __init__(self, DT, Q, R=R, alpha=ALPHA, beta=BETA, kappa=KAPPA):
		self.R = R
		self.alpha = alpha
		self.beta = beta
		self.kappa = kappa
		super().__init__(DT, Q)

	def setup_ukf(self, nx):
		lamb = self.alpha ** 2 * (nx + self.kappa) - nx
		# calculate weights
		wm = [lamb / (lamb + nx)]
		wc = [(lamb / (lamb + nx)) + (1 - self.alpha ** 2 + self.beta)]
		for i in range(2 * nx):
			wm.append(1.0 / (2 * (nx + lamb)))
			wc.append(1.0 / (2 * (nx + lamb)))
		gamma = math.sqrt(nx + lamb)

		wm = np.array([wm])
		wc = np.array([wc])

		return wm, wc, gamma


	def generate_sigma_points(self, xEst, PEst, gamma):
		sigma = xEst
		Psqrt = scipy.linalg.sqrtm(PEst)
		n = len(xEst[:, 0])
		# Positive direction
		for i in range(n):
			sigma = np.hstack((sigma, xEst + gamma * Psqrt[:, i:i + 1]))

		# Negative direction
		for i in range(n):
			sigma = np.hstack((sigma, xEst - gamma * Psqrt[:, i:i + 1]))

		return sigma


	def predict_sigma_motion(self, sigma, u):
		"""
			Sigma Points prediction with motion model
		"""
		for i in range(sigma.shape[1]):
			sigma[:, i:i + 1] = self.motion_model(sigma[:, i:i + 1], u)

		return sigma


	def predict_sigma_observation(self, sigma):
		"""
			Sigma Points prediction with observation model
		"""
		for i in range(sigma.shape[1]):
			sigma[0:2, i] = self.observation_model(sigma[:, i])

		sigma = sigma[0:2, :]

		return sigma


	def calc_sigma_covariance(self, x, sigma, wc, Pi):
		nSigma = sigma.shape[1]
		d = sigma - x[0:sigma.shape[0]]
		P = Pi
		for i in range(nSigma):
			P = P + wc[0, i] * d[:, i:i + 1] @ d[:, i:i + 1].T
		return P


	def calc_pxz(self, sigma, x, z_sigma, zb, wc):
		nSigma = sigma.shape[1]
		dx = sigma - x
		dz = z_sigma - zb[0:2]
		P = np.zeros((dx.shape[0], dz.shape[0]))

		for i in range(nSigma):
			P = P + wc[0, i] * dx[:, i:i + 1] @ dz[:, i:i + 1].T

		return P

	def estimation(self, xEst, PEst, z, u, wm, wc, gamma):

		#  Predict
		sigma = self.generate_sigma_points(xEst, PEst, gamma)
		sigma = self.predict_sigma_motion(sigma, u)
		xPred = (wm @ sigma.T).T
		PPred = self.calc_sigma_covariance(xPred, sigma, wc, self.Q)

		#  Update
		zPred = self.observation_model(xPred)
		y = z - zPred
		sigma = self.generate_sigma_points(xPred, PPred, gamma)
		zb = (wm @ sigma.T).T
		z_sigma = self.predict_sigma_observation(sigma)
		st = self.calc_sigma_covariance(zb, z_sigma, wc, self.R)
		Pxz = self.calc_pxz(sigma, xPred, z_sigma, zb, wc)
		K = Pxz @ np.linalg.inv(st)
		xEst = xPred + K @ y
		PEst = PPred - K @ st @ K.T

		return xEst, PEst



def main():
	print(__file__ + " start!!")

	nx = 4  # State Vector [x y yaw v]'
	xEst = np.zeros((nx, 1))
	xTrue = np.zeros((nx, 1))
	PEst = np.eye(nx)
	xDR = np.zeros((nx, 1))  # Dead reckoning

	Filter = UnscentedKF(DT, Q, R)
	wm, wc, gamma = Filter.setup_ukf(nx)

	history = History(xEst, xTrue)

	time = 0.0

	while SIM_TIME >= time:
		time += DT
		u = simulator.calc_input()

		# SLAM
		xTrue, z, xDR, ud = Filter.observation(xTrue, xDR, u)
		xEst, PEst = Filter.estimation(xEst, PEst, z, ud, wm, wc, gamma)

		history.update(xEst, xDR, xTrue, z)

		if show_animation:
			history.plot(xEst, PEst)


if __name__ == '__main__':
    main()
