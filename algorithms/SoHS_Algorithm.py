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

        self.score_ = np.empty([1, 1])
        self.point_ = np.empty([0, 3])

    def fit(self):
        """
        Squares of Heights Search Algorithm fit
        :return: None
        """

        zoneL, zones = self.__pc_decomposition()
        self.score_ = np.empty(zones.shape[:2])

        soh = np.empty([0, 3])

        for row in range(self.score_.shape[0]):
            for col in range(self.score_.shape[1]):
                self.score_[row, col] = self.__zone_estimate(zones[row, col])

                if self.score_[row, col] is not None:
                    flat = zones[row, col].transpose(2, 0, 1).reshape(3, -1)
                    flat = np.array([point for point in zip(flat[0], flat[1], flat[2])])
                    point = np.mean(flat, axis=0)

                    soh = np.vstack([soh, point])
                    self.inliers_ = np.vstack([self.inliers_, flat])

        self.area_ = self.__zone_area(in_pc=soh, in_zoneL=zoneL)
        self.point_ = self.__point_determination(in_pc=soh)

        return None

    def __pc_decomposition(self):
        """
        Point Cloud Decomposition
        :return: L zone shape, Squares of heights
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

        return zoneL, frame_zones

    def __zone_estimate(self, in_pc):
        """
        Zone estimate calculation
        :param in_pc: Input point cloud
        :return: Point cloud variance
        """

        height_flat = in_pc[:, :, 2].flatten()

        diff_h = np.max(height_flat) - np.min(height_flat)
        mean_h = np.mean(height_flat)
        std_h = np.std(height_flat)

        var_h = std_h / mean_h if diff_h <= 0.5 or std_h <= 0.1 else None

        return var_h

    def __zone_area(self, in_pc, in_zoneL):
        """
        Zone area calculation
        :param in_pc: Input point cloud
        :return: Zone area
        """

        square = (self.scale_ ** 2) * np.prod(in_zoneL)
        area = len(in_pc) * square

        return area

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

    print(f"CDS algorithm result:\n\t- Point: {alg.point_}\n\t- Area: {alg.area_}")
    print(f"Time: {stop_time}")
