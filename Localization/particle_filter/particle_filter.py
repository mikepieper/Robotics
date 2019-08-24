"""

Particle Filter localization sample

author: Atsushi Sakai (@Atsushi_twi)

"""

import numpy as np
import math
import matplotlib.pyplot as plt

from pf_history import PfHistory
from utils import gauss_likelihood

import sys
sys.path.append('..')
from localization import BayesFilter
from simulator import Simulator


simulator = Simulator()


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

class ParticleFilter(BayesFilter):

    def __init__(self, DT):
        super().__init__(DT)

    def observation(self, xTrue, xd, u, RFID):
        xTrue = self.motion_model(xTrue, u)

        # add noise to gps x-y
        z = np.zeros((0, 3))

        for i in range(len(RFID[:, 0])):

            dx = xTrue[0, 0] - RFID[i, 0]
            dy = xTrue[1, 0] - RFID[i, 1]
            d = math.sqrt(dx**2 + dy**2)
            if d <= MAX_RANGE:
                dn = d + np.random.randn() * Qsim[0, 0]  # add noise
                zi = np.array([[dn, RFID[i, 0], RFID[i, 1]]])
                z = np.vstack((z, zi))

        # add noise to input
        ud1 = u[0, 0] + np.random.randn() * Rsim[0, 0]
        ud2 = u[1, 0] + np.random.randn() * Rsim[1, 1]
        ud = np.array([[ud1, ud2]]).T

        xd = self.motion_model(xd, ud)

        return xTrue, z, xd, ud


    def calc_covariance(self, xEst, px, pw):
        cov = np.zeros((3, 3))

        for i in range(px.shape[1]):
            dx = (px[:, i] - xEst)[0:3]
            cov += pw[0, i] * dx.dot(dx.T)

        return cov


    def localization(self, px, pw, xEst, PEst, z, u):
        """
        Localization with Particle filter
        """

        for ip in range(NP):
            x = np.array([px[:, ip]]).T
            w = pw[0, ip]
            #  Predict with random input sampling
            ud1 = u[0, 0] + np.random.randn() * Rsim[0, 0]
            ud2 = u[1, 0] + np.random.randn() * Rsim[1, 1]
            ud = np.array([[ud1, ud2]]).T
            x = self.motion_model(x, ud)

            #  Calc Importance Weight
            for i in range(len(z[:, 0])):
                dx = x[0, 0] - z[i, 1]
                dy = x[1, 0] - z[i, 2]
                prez = math.sqrt(dx**2 + dy**2)
                dz = prez - z[i, 0]
                w = w * gauss_likelihood(dz, math.sqrt(Q[0, 0]))

            px[:, ip] = x[:, 0]
            pw[0, ip] = w

        pw = pw / pw.sum()  # normalize

        xEst = px.dot(pw.T)
        PEst = self.calc_covariance(xEst, px, pw)

        px, pw = resampling(px, pw)

        return xEst, PEst, px, pw






def resampling(px, pw):
    """
    low variance re-sampling
    """

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



def main():
    print(__file__ + " start!!")

    time = 0.0

    # RFID positions [x, y]
    RFID = np.array([[10.0, 0.0],
                     [10.0, 10.0],
                     [0.0, 15.0],
                     [-5.0, 20.0]])

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)
    xDR = np.zeros((4, 1))  # Dead reckoning

    px = np.zeros((4, NP))  # Particle store
    pw = np.zeros((1, NP)) + 1.0 / NP  # Particle weight
    
    Filter = ParticleFilter(DT)

    history = PfHistory(xEst, xTrue)


    while SIM_TIME >= time:
        time += DT
        u = simulator.calc_input()

        xTrue, z, xDR, ud = Filter.observation(xTrue, xDR, u, RFID)

        xEst, PEst, px, pw = Filter.localization(px, pw, xEst, PEst, z, ud)

        history.update(xEst, xDR, xTrue)

        if show_animation:
            history.plot(xEst, PEst, xTrue, z, RFID, px)


if __name__ == '__main__':
    main()
