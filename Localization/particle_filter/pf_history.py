import numpy as np
import math
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from history import History



class PfHistory(History):

	def __init__(self, xEst, xTrue):
		super().__init__(xEst, xTrue)

	def plot(self, xEst, PEst, xTrue, z, RFID, px):
		plt.cla()

		for i in range(len(z[:, 0])):
			plt.plot([xTrue[0, 0], z[i, 1]], [xTrue[1, 0], z[i, 2]], "-k")

		plt.plot(RFID[:, 0], RFID[:, 1], "*k")
		plt.plot(px[0, :], px[1, :], ".r")
		
		plt.plot(np.array(self.hxTrue[0, :]).flatten(),
					np.array(self.hxTrue[1, :]).flatten(), "-b")
		plt.plot(np.array(self.hxDR[0, :]).flatten(),
					np.array(self.hxDR[1, :]).flatten(), "-k")
		plt.plot(np.array(self.hxEst[0, :]).flatten(),
					np.array(self.hxEst[1, :]).flatten(), "-r")
		self.plot_covariance_ellipse(xEst, PEst)
		plt.axis("equal")
		plt.grid(True)
		plt.pause(0.001)
