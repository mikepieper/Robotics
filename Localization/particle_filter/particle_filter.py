"""

Particle Filter localization sample

author: Atsushi Sakai (@Atsushi_twi)

"""

import numpy as np
import math
import matplotlib.pyplot as plt

from history import History
from utils import gauss_likelihood, low_variance_resampling, calculate_covariance

import sys
sys.path.append('..')
from simulator import ControlModel


# Estimation parameter of PF
Q = np.diag([0.1])**2  # range error
R = np.diag([1.0, np.deg2rad(40.0)])**2  # input error

#  Simulation parameter
Qsim = np.diag([0.2])**2
Rsim = np.diag([1.0, np.deg2rad(30.0)])**2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range

# Particle filter parameter
NP = 100  # Number of Particle
NTh = NP / 2.0  # Number of particle for re-sampling

show_animation = True

class MotionModel():
	
	def __init__(self):
		pass

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

   
class ParticleFilter():

	def __init__(self, motion_model, DT, Q, R):
		self.motion_model = motion_model
		self.DT = DT
		self.Q = Q
		self.R = R


	# Table 4.3
	def __call__(self, px, pw, mu, sigma, z, u):
		for i in range(NP): # line 3
			x_past = np.array([px[:, i]]).T
			w = pw[0, i]
			x = self.sample(u, x_past) # line 4

			#  Calc Importance Weight
			w = pw[0,i]
			w = self.calculate_importance_weight(z, x, w) # line 5

			# line 7
			px[:, i] = x[:, 0]
			pw[0, i] = w

		# lines 8-11
		pw = pw / pw.sum()  # normalize
		mu = px.dot(pw.T)
		sigma = self.calculate_covariance(mu, px, pw)
		px, pw = self.low_variance_resampling(px, pw)

		return mu, sigma, px, pw

	# line 4
	def sample(self, u, x_past):
		# add noise to input
		u1 = u[0, 0] + np.random.randn() * Rsim[0, 0]
		u2 = u[1, 0] + np.random.randn() * Rsim[1, 1]
		u = np.array([[u1, u2]]).T
		
		#  Predict with random input sampling
		x = self.motion_model(u, x_past)
		return x

	# line 5
	def calculate_importance_weight(self, z, x, w):
		for i in range(len(z[:, 0])):
			dx = x[0, 0] - z[i, 1]
			dy = x[1, 0] - z[i, 2]
			prez = math.sqrt(dx**2 + dy**2)
			dz = prez - z[i, 0]
			w = w * gauss_likelihood(dz, math.sqrt(self.Q[0, 0]))

		return w

	def calculate_covariance(self, mu, px, pw):
		cov = np.zeros((3, 3))

		for i in range(px.shape[1]):
			dx = (px[:, i] - mu)[0:3]
			cov += pw[0, i] * dx.dot(dx.T)

		return cov


	def low_variance_resampling(self, px, pw):
		Neff = 1.0 / (pw.dot(pw.T))[0, 0]  # Effective particle number
		if Neff < NTh:
			wcum = np.cumsum(pw)
			base = np.cumsum(pw * 0.0 + 1 / NP) - 1 / NP
			resampleid = base + np.random.rand(base.shape[0]) / NP

			inds = []
			ind = 0
			for ip in range(NP):
				while resampleid[ip] > wcum[ind]:
					ind += 1
				inds.append(ind)

			px = px[:, inds]
			pw = np.zeros((1, NP)) + 1.0 / NP  # init weight

		return px, pw


def observation(motion_model, pose, mu, u, RFID):

    pose = motion_model(u, pose)

    # add noise to gps x-y
    z = np.zeros((0, 3))

    for i in range(len(RFID[:, 0])):

        dx = pose[0, 0] - RFID[i, 0]
        dy = pose[1, 0] - RFID[i, 1]
        d = math.sqrt(dx**2 + dy**2)
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Qsim[0, 0]  # add noise
            zi = np.array([[dn, RFID[i, 0], RFID[i, 1]]])
            z = np.vstack((z, zi))

    # add noise to input
    u1 = u[0, 0] + np.random.randn() * Rsim[0, 0]
    u2 = u[1, 0] + np.random.randn() * Rsim[1, 1]
    u = np.array([[u1, u2]]).T

    mu = motion_model(u, mu)

    return pose, z, mu, u

def main():
    print(__file__ + " start!!")

    time = 0.0

    # RFID positions [x, y]
    RFID = np.array([[10.0, 0.0],
                     [10.0, 10.0],
                     [0.0, 15.0],
                     [-5.0, 20.0]])

    # State Vector [x y yaw v]'
    mu = np.zeros((4, 1))
    pose = np.zeros((4, 1))
    sigma = np.eye(4)

    px = np.zeros((4, NP))  # Particle store
    pw = np.zeros((1, NP)) + 1.0 / NP  # Particle weight
    xDR = np.zeros((4, 1))  # Dead reckoning, mu
    
    control_model = ControlModel()
    motion_model = MotionModel()
    Localization = ParticleFilter(motion_model, DT, Q, R)
    history = History(mu, pose)


    while SIM_TIME >= time:
        time += DT
        u = control_model()

        pose, z, xDR, ud = observation(motion_model, pose, xDR, u, RFID)

        mu, sigma, px, pw = Localization(px, pw, mu, sigma, z, ud)

        history.update(mu, xDR, pose)

        if show_animation:
            history.plot(mu, sigma, pose, z, RFID, px)


if __name__ == '__main__':
    main()
