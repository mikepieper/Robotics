"""

Histogram Filter 2D localization example


In this simulation, x,y are unknown, yaw is known.

Initial position is not needed.

author: Atsushi Sakai (@Atsushi_twi)

"""

import copy
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from grid_map import GridMap

import sys
sys.path.append('..')
from localization import BayesFilter
from simulator import Simulator

simulator = Simulator()

# Parameters
EXTEND_AREA = 10.0  # [m] grid map extended length
SIM_TIME = 50.0  # simulation time [s]
DT = 0.1  # time tick [s]
MAX_RANGE = 10.0  # maximum observation range
MOTION_STD = 1.0  # standard deviation for motion gaussian distribution
RANGE_STD = 3.0  # standard deviation for observation gaussian distribution


# simulation parameters
NOISE_RANGE = 2.0  # [m] 1σ range noise parameter
NOISE_SPEED = 0.5  # [m/s] 1σ speed noise parameter


show_animation = True

class HistogramFilter(BayesFilter):
	
	def __init__(self, DT):
		super().__init__(DT)


	def observation(self, xTrue, u, RFID):
		xTrue = self.motion_model(xTrue, u)
		z = np.zeros((0, 3))
		for i in range(len(RFID[:, 0])):
			dx = xTrue[0, 0] - RFID[i, 0]
			dy = xTrue[1, 0] - RFID[i, 1]
			d = math.sqrt(dx**2 + dy**2)
			if d <= MAX_RANGE:
				# add noise to range observation
				dn = d + np.random.randn() * NOISE_RANGE
				zi = np.array([dn, RFID[i, 0], RFID[i, 1]])
				z = np.vstack((z, zi))

		# add noise to speed
		ud = u[:, :]
		ud[0] += np.random.randn() * NOISE_SPEED

		return xTrue, z, ud


	def localization(self, grid_map, u, z, yaw):
		grid_map = self.motion_update(grid_map, u, yaw)
		grid_map = self.observation_update(grid_map, z, RANGE_STD)
		return grid_map


	def motion_update(self, grid_map, u, yaw):
		grid_map.dx += DT * math.cos(yaw) * u[0]
		grid_map.dy += DT * math.sin(yaw) * u[0]

		x_shift = grid_map.dx // grid_map.xy_reso
		y_shift = grid_map.dy // grid_map.xy_reso

		if abs(x_shift) >= 1.0 or abs(y_shift) >= 1.0:  # map should be shifted
			grid_map = self.map_shift(grid_map, int(x_shift), int(y_shift))
			grid_map.dx -= x_shift * grid_map.xy_reso
			grid_map.dy -= y_shift * grid_map.xy_reso

		grid_map.data = gaussian_filter(grid_map.data, sigma=MOTION_STD)

		return grid_map


	def observation_update(self, gmap, z, std):
		for iz in range(z.shape[0]):
			for ix in range(gmap.xw):
				for iy in range(gmap.yw):
					gmap.data[ix][iy] *= self.calc_gaussian_observation_pdf(
						gmap, z, iz, ix, iy, std)

		gmap.normalize_probability()
		return gmap


	def calc_gaussian_observation_pdf(self, gmap, z, iz, ix, iy, std):

		# predicted range
		x = ix * gmap.xy_reso + gmap.minx
		y = iy * gmap.xy_reso + gmap.miny
		d = math.sqrt((x - z[iz, 1])**2 + (y - z[iz, 2])**2)

		# likelihood
		pdf = (1.0 - norm.cdf(abs(d - z[iz, 0]), 0.0, std))

		return pdf


	def map_shift(self, grid_map, x_shift, y_shift):
		tgmap = copy.deepcopy(grid_map.data)

		for ix in range(grid_map.xw):
			for iy in range(grid_map.yw):
				nix = ix + x_shift
				niy = iy + y_shift

				if 0 <= nix < grid_map.xw and 0 <= niy < grid_map.yw:
					grid_map.data[ix + x_shift][iy + y_shift] = tgmap[ix][iy]

		return grid_map



def main():
	print(__file__ + " start!!")

	# RFID positions [x, y]
	RFID = np.array([[10.0, 0.0],
						[10.0, 10.0],
						[0.0, 15.0],
						[-5.0, 20.0]])

	time = 0.0

	Filter = HistogramFilter(DT)

	xTrue = np.zeros((4, 1))
	grid_map = GridMap()

	while SIM_TIME >= time:
		time += DT
		print("Time:", time)

		u = simulator.calc_input()

		yaw = xTrue[2, 0]  # Orientation is known
		xTrue, z, ud = Filter.observation(xTrue, u, RFID)

		grid_map = Filter.localization(grid_map, u, z, yaw)

		if show_animation:
			grid_map.plot(xTrue, RFID, z, time)

	print("Done")


if __name__ == '__main__':
    main()
