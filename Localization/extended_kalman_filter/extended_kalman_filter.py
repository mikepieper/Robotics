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

simulator = Simulator()

# Covariance for EKF simulation
Q = np.diag([
    0.1, # variance of location on x-axis
    0.1, # variance of location on y-axis
    np.deg2rad(1.0), # variance of yaw angle
    1.0 # variance of velocity
    ])**2  # predict state covariance
R = np.diag([1.0, 1.0])**2  # Observation x,y position covariance


DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]

class EKF(KalmanFilter):
	
	def __init__(self, DT, Q, R):
		super().__init__(DT, Q, R)


	def jacobF(self, x, u):
		"""
		Jacobian of Motion Model

		motion model
		x_{t+1} = x_t+v*dt*cos(yaw)
		y_{t+1} = y_t+v*dt*sin(yaw)
		yaw_{t+1} = yaw_t+omega*dt
		v_{t+1} = v{t}
		so
		dx/dyaw = -v*dt*sin(yaw)
		dx/dv = dt*cos(yaw)
		dy/dyaw = v*dt*cos(yaw)
		dy/dv = dt*sin(yaw)
		"""
		yaw = x[2, 0]
		v = u[0, 0]
		jF = np.array([
			[1.0, 0.0, -self.DT * v * math.sin(yaw), self.DT * math.cos(yaw)],
			[0.0, 1.0, self.DT * v * math.cos(yaw), self.DT * math.sin(yaw)],
			[0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 1.0]])

		return jF


	def jacobH(self, x):
		# Jacobian of Observation Model
		jH = np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0]
		])

		return jH


	def estimation(self, xEst, PEst, z, u):

		#  Predict
		xPred = self.motion_model(xEst, u)
		jF = self.jacobF(xPred, u)
		PPred = jF@PEst@jF.T + self.Q

		#  Update
		jH = self.jacobH(xPred)
		zPred = self.observation_model(xPred)
		y = z - zPred
		S = jH@PPred@jH.T + self.R
		K = PPred@jH.T@np.linalg.inv(S)
		xEst = xPred + K@y
		PEst = (np.eye(len(xEst)) - K@jH)@PPred

		return xEst, PEst



def main():
	print(__file__ + " start!!")

	time = 0.0

	# State Vector [x y yaw v]'
	xEst = np.zeros((4, 1))
	xTrue = np.zeros((4, 1))
	PEst = np.eye(4)
	xDR = np.zeros((4, 1))  # Dead reckoning
	
	Filter = EKF(DT, Q, R)

	history = History(xEst, xTrue)


	while SIM_TIME >= time:
		time += DT
		u = simulator.calc_input()

		xTrue, z, xDR, ud = Filter.observation(xTrue, xDR, u)
		xEst, PEst = Filter.estimation(xEst, PEst, z, ud)

		history.update(xEst, xDR, xTrue, z)

		if show_animation:
			history.plot(xEst, PEst)


if __name__ == '__main__':
    main()
