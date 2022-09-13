import time

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error

import ALG_Library as lib
from tests.PC_Generation import PC_gen


class CDS_alg:
    """
    Cluster Deviation Search Algorithm
    """

    def __init__(self, in_data=np.empty([0, 3]), in_scale=0.01):
        self.data_ = in_data.copy()
        self.scale_ = in_scale

        self.inliers_ = np.empty([0, 3])
        self.equation_ = np.empty([0, 4])

        self.area_ = 0.0
        self.slope_ = 0.0
        self.deviation_ = 0.0

        self.score_ = 0.0
        self.point_ = np.empty([0, 3])

    def fit(self):
        """
        Cluster Deviation Search Algorithm fit
        :return: None
        """

        point_cloud = self.data_.copy()

        clusters = self.__zone_clustering(in_pc=point_cloud)
        for cluster in clusters:
            equation, slope = self.__zone_slope(in_pc=cluster)
            area = self.__zone_area(in_pc=cluster)
            deviation = self.__zone_deviation(in_pc=cluster, in_eq=equation)

            score = self.__zone_estimate(in_pc=cluster, in_slope=slope, in_area=area, in_dev=deviation)
            #print(f">> Equation: {equation}; Slope: {slope}; Area: {area}; Deviation: {deviation}. Score: {score}.")

            self.deviation_ = deviation
            if score > self.score_:
                self.inliers_ = cluster
                self.equation_ = equation
                self.slope_ = slope
                self.area_ = area
                self.score_ = score
                self.point_ = self.__point_determination(in_pc=cluster)

        return None

    def __zone_clustering(self, in_pc):
        """
        Zones clustering in point cloud with Kd-tree and Euclidean clustering algorithms
        :param in_pc: Input point cloud
        :return: Zones points clusters
        """

        # NEED TO OPTIMIZE
        model = DBSCAN(
            eps=self.scale_ * 2.0,
            min_samples=9,
            metric='euclidean',
            algorithm='ball_tree',
            leaf_size=30
        )
        model.fit(X=in_pc)

        num_clusters = set(model.labels_)

        clusters = []
        for num in num_clusters:
            if num == -1: continue

            clusters.append(
                np.array([
                    in_pc[i]
                    for i in range(len(model.labels_))
                    if model.labels_[i] == num
                ])
            )

        return clusters

    def __zone_area(self, in_pc):
        """
        Zone area calculation
        :param in_pc: Input point cloud
        :return: Zone area
        """

        square = self.scale_ ** 2
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

    def __zone_deviation(self, in_pc, in_eq):
        """
        Zone deviation calculation with MSE
        :param in_pc: Input point cloud
        :param in_eq: Input zone equation
        :return: Zone deviation
        """
        deviation = mean_squared_error(
            np.zeros([len(in_pc)]),
            np.sum(np.c_[in_pc, np.ones(len(in_pc))] * np.array([in_eq for _ in range(len(in_pc))]), axis=1)
        )

        return deviation

    def __zone_estimate(self, in_pc, in_slope, in_area, in_dev):
        """
        Zone estimate calculation
        :param in_slope:
        :param in_area:
        :param in_dev:
        :return:
        """

        area_cl, deviation_cl, slope_cl = 0.0, 0.0, 0.0
        if in_area >= 0.126 and in_dev <= 0.05 and in_slope <= 15.0:
            area_cl = in_area / self.__zone_area(in_pc=in_pc)
            deviation_cl = in_dev / len(in_pc)
            slope_cl = in_slope / 15.0

        est = area_cl * (1 - 0.5 * (deviation_cl + slope_cl))

        return est

    def __point_determination(self, in_pc):
        """
        Landing point determination with geometric median
        :param in_pc: Input point cloud
        :return: Point coordinates
        """

        g_median = np.array(lib.geometric_median(in_pc))

        return g_median


if __name__ == '__main__':
    cloud_shape = np.array([640, 480])
    cloud_step = 0.00625

    gen = PC_gen(shape=cloud_shape, step=cloud_step)
    cloud = gen.plane_gen(hiegh=0.5, noise=0.0, slope=00.0, loss=0.0)
    gen.visualization(cloud=cloud)

    time_start = time.time()
    alg = CDS_alg(in_data=cloud, in_scale=cloud_step)
    alg.fit()
    stop_time = time.time() - time_start

    print(f"CDS algorithm result:\n\t- Point: {alg.point_}\n\t- Area: {alg.area_}\n\t- Slope: {alg.slope_}\n\t- Deviation: {alg.deviation_}")
    print(f"Time: {stop_time}")
