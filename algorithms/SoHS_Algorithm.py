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
        self.equation_ = np.empty([0, 4])

        self.area_ = 0.0
        self.slope_ = 0.0
        self.deviation_ = 0.0

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
        soh_dev = np.empty([0])

        for row in range(self.score_.shape[0]):
            for col in range(self.score_.shape[1]):
                self.score_[row, col] = self.__zone_estimate(zones[row, col])

                if np.isnan(self.score_[row, col]) == False:
                    flat = zones[row, col].transpose(2, 0, 1).reshape(3, -1)
                    flat = np.array([point for point in zip(flat[0], flat[1], flat[2])])
                    point = np.mean(flat, axis=0)

                    soh = np.vstack([soh, point])
                    soh_dev = np.append(soh_dev, self.score_[row, col])
                    self.inliers_ = np.vstack([self.inliers_, flat])

        area = self.__zone_area(in_pc=soh, in_zoneL=zoneL)
        equation, slope = self.__zone_slope(in_pc=soh)
        deviation = np.mean(soh_dev, axis=0)

        #print(f">> Equation: {equation}; Slope: {slope}; Area: {area}; Deviation: {deviation}.")

        self.deviation_ = deviation
        if area >= 0.126 and slope <= 15.0 and deviation <= 0.05:
            self.equation_ = equation
            self.area_ = area
            self.slope_ = slope
            self.point_ = self.__point_determination(in_pc=soh)

        return None

    def __pc_decomposition(self):
        """
        Point Cloud Decomposition
        :return: L zone shape, Squares of heights
        """

        # NEED TO OPTIMIZE
        zoneL = lib.divide_and_conquer(shape=self.shape_.copy())
        zoneS = np.divide(self.shape_, zoneL)

        while np.prod(zoneS) < 100.0:
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

        var_h = std_h / mean_h if diff_h <= 0.5 or std_h <= 0.1 else np.nan

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

    def __zone_slope(self, in_pc):
        """
        Zone equation and slope calculation
        (based on https://www.programcreek.com/python/?CodeExample=fit+plane)
        :param in_pc: Input point cloud
        :return: Zone equation, Zone slope
        """

        mean = np.mean(in_pc, axis=0)
        xyz_c = in_pc - mean[None, :]
        l, v = np.linalg.eig(xyz_c.T.dot(xyz_c))
        abc = v[:, np.argmin(l)]
        d = -np.sum(abc * mean)

        equation = np.r_[abc, d] / np.linalg.norm(abc)
        if equation[2] < 0:
            equation = np.array([equation[1], equation[0], -equation[2], -equation[3]])

        # rad_phi = (a, b) / (magnitude(a) * magnitude(b))
        # grad_phi = abs(arccos(rad_phi) * 180 / PI)
        normal = np.array([0.0, 0.0, 1.0])
        slope = np.abs(
            np.arccos(
                np.dot(normal, equation[:3]) /
                (np.linalg.norm(normal) * np.linalg.norm(equation[:3]))
            ) * 180.0 / np.pi
        )

        return equation, slope

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
    cloud_step = 0.00625

    gen = PC_gen(shape=cloud_shape, step=cloud_step)
    cloud = gen.plane_gen(hiegh=0.5, noise=0.0, slope=00.0, loss=0.0)
    gen.visualization(cloud=cloud)

    time_start = time.time()
    alg = SoHS_alg(in_data=cloud, in_shape=cloud_shape, in_scale=cloud_step)
    alg.fit()
    stop_time = time.time() - time_start

    print(f"SoHS algorithm result:\n\t- Point: {alg.point_}\n\t- Area: {alg.area_}\n\t- Slope: {alg.slope_}\n\t- Deviation: {alg.deviation_}")
    print(f"Time: {stop_time}")
