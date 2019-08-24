import numpy as np
import matplotlib.pyplot as plt

XY_RESO = 0.5  # xy grid resolution
MINX = -15.0
MINY = -5.0
MAXX = 15.0
MAXY = 25.0

class GridMap():

    def __init__(self, xy_reso=XY_RESO, minx=MINX, miny=MINY, maxx=MAXX, maxy=MAXY):
        self.xy_reso = xy_reso
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.xw = int(round((self.maxx - self.minx) / self.xy_reso))
        self.yw = int(round((self.maxy - self.miny) / self.xy_reso))
        self.dx = 0.0  # movement distance
        self.dy = 0.0  # movement distance
        self.data = [[1.0 for _ in range(self.yw)] for _ in range(self.xw)]
        self.normalize_probability()
        self.calc_grid_index()


    def normalize_probability(self):
        sump = sum([sum(igmap) for igmap in self.data])

        for ix in range(self.xw):
            for iy in range(self.yw):
                self.data[ix][iy] /= sump

    def calc_grid_index(self):
        self.mx, self.my = np.mgrid[slice(self.minx - self.xy_reso / 2.0, self.maxx + self.xy_reso / 2.0, self.xy_reso),
                        slice(self.miny - self.xy_reso / 2.0, self.maxy + self.xy_reso / 2.0, self.xy_reso)]

    def plot(self, xTrue, RFID, z, time):
        plt.cla()
        self.draw_heat_map()
        plt.plot(xTrue[0, :], xTrue[1, :], "xr")
        plt.plot(RFID[:, 0], RFID[:, 1], ".k")
        for i in range(z.shape[0]):
            plt.plot([xTrue[0, :], z[i, 1]], [
                        xTrue[1, :], z[i, 2]], "-k")
        plt.title("Time[s]:" + str(time)[0: 4])
        plt.pause(0.1)

    def draw_heat_map(self):
        maxp = max([max(igmap) for igmap in self.data])
        plt.pcolor(self.mx, self.my, self.data, vmax=maxp, cmap=plt.cm.get_cmap("Blues"))
        plt.axis("equal")