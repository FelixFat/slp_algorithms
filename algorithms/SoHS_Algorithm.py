import time

import numpy as np

import ALG_Library as lib
from tests.PC_Generation import PC_gen


class SoHS_alg:
    """
    Squares of Heights Search Algorithm
    """

    def __init__(self, in_data=np.empty([0, 3]), in_shape=np.array([100, 100]), in_scale=0.01):
        self.data_ = in_data.copy()
        self.shape_ = in_shape.astype(int)
        self.scale_ = in_scale

        self.inliers_ = np.empty([0, 3])

        self.area_ = 0.0
        self.deviation_ = 0.0

        self.score_ = 0.0
        self.point_ = np.empty([0, 3])

    def fit(self):
        self.__pc_decomposition()
        return None

    def __pc_decomposition(self):
        """
        Point Cloud Decomposition
        :return: Squares of heights
        """

        zoneL = lib.divide_and_conquer(shape=self.shape_.copy())
        zoneS = np.divide(self.shape_, zoneL)

        while zoneL[0] > 40 and zoneL[1] > 40:
            zoneL = zoneL / 2.0
            zoneS = zoneS * 2.0
        zoneL, zoneS = zoneL.astype(int), zoneS.astype(int)

        frame = np.empty([self.shape_[1], self.shape_[0], 3])
        frame_zones = np.empty([zoneS[1], zoneS[0], zoneL[1], zoneL[0], 3])

        for i in range(len(self.data_)):
            w, h = i // self.shape_[0], i % self.shape_[0]
            frame[w, h] = self.data_[i]

        for row_i in range(zoneS[1]):
            for col_i in range(zoneS[0]):
                w_start, w_stop = row_i * zoneL[1], row_i * zoneL[1] + zoneL[1]
                h_start, h_stop = col_i * zoneL[0], col_i * zoneL[0] + zoneL[0]
                frame_zones[row_i, col_i] = frame[w_start:w_stop, h_start:h_stop]

        return frame_zones

    def __pc_estimation(self):
        return 0.0

    def __zone_estimation(self, in_pc):
        return 0.0

    def __soh_organization(self, in_pc):
        return 0.0

    def __point_determination(self, in_pc):
        """
        Landing point determination with geometric median
        :param in_pc: Input point cloud
        :return: Point coordinates
        """

        point_mean = np.mean(in_pc, axis=0)

        return point_mean


if __name__ == '__main__':
    cloud_shape = np.array([640, 480])
    cloud_step = 0.01

    gen = PC_gen(shape=cloud_shape, step=cloud_step)
    cloud = gen.plane_gen(hiegh=0.5, noise=0.001, loss=0.0)

    time_start = time.time()
    alg = SoHS_alg(in_data=cloud, in_shape=cloud_shape, in_scale=cloud_step)
    alg.fit()
    stop_time = time.time() - time_start

    print(f"CDS algorithm result:\n\t- Point: {alg.point_}\n\t- Area: {alg.area_}\n\t- Deviation: {alg.deviation_}")
    print(f"Time: {stop_time}")
