import numpy as np
import math
import matplotlib.pyplot as plt

class History():

	def __init__(self, xEst, xTrue):
		# history
		self.hxEst = xEst
		self.hxTrue = xTrue
		self.hxDR = xTrue
		self.hz = np.zeros((2, 1))

	def update(self, xEst, xDR, xTrue, z=None):
		self.hxEst = np.hstack((self.hxEst, xEst))
		self.hxDR = np.hstack((self.hxDR, xDR))
		self.hxTrue = np.hstack((self.hxTrue, xTrue))
		if z:
			self.hz = np.hstack((self.hz, z))

	def plot(self, xEst, PEst):
		plt.cla()
		plt.plot(self.hz[0, :], self.hz[1, :], ".g")
		plt.plot(self.hxTrue[0, :].flatten(),
					self.hxTrue[1, :].flatten(), "-b")
		plt.plot(self.hxDR[0, :].flatten(),
					self.hxDR[1, :].flatten(), "-k")
		plt.plot(self.hxEst[0, :].flatten(),
					self.hxEst[1, :].flatten(), "-r")
		self.plot_covariance_ellipse(xEst, PEst)
		plt.axis("equal")
		plt.grid(True)
		plt.pause(0.001)

	def plot_covariance_ellipse(self, xEst, PEst):  # pragma: no cover
		Pxy = PEst[0:2, 0:2]
		eigval, eigvec = np.linalg.eig(Pxy)

		if eigval[0] >= eigval[1]:
			bigind = 0
			smallind = 1
		else:
			bigind = 1
			smallind = 0

		t = np.arange(0, 2 * math.pi + 0.1, 0.1)
		
		# eigval[bigind] or eiqval[smallind] were occassionally negative numbers extremely
		# close to 0 (~10^-20), catch these cases and set the respective variable to 0
		try:
			a = math.sqrt(eigval[bigind])
		except ValueError:
			a = 0

		try:
			b = math.sqrt(eigval[smallind])
		except ValueError:
			b = 0
		
		x = [a * math.cos(it) for it in t]
		y = [b * math.sin(it) for it in t]
		angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
		R = np.array([[math.cos(angle), math.sin(angle)],
					[-math.sin(angle), math.cos(angle)]])
		fx = R@(np.array([x, y]))
		px = np.array(fx[0, :] + xEst[0, 0]).flatten()
		py = np.array(fx[1, :] + xEst[1, 0]).flatten()
		plt.plot(px, py, "--r")