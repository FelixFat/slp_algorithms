import time
import warnings

import numpy as np
from sklearn.cluster import DBSCAN

import ALG_Library as lib
from tests.PC_Generation import PC_gen

class FPS_alg:
    """
    Flat Plane Search Algorithm
    """

    def __init__(self, in_data=np.empty([0, 3]), in_scale=0.01):
        self.data_ = in_data.copy()
        self.scale_ = in_scale

        self.inliers_ = np.empty([0, 3])
        self.equation_ = np.empty([0, 4])

        self.slope_ = 0.0
        self.area_ = 0.0

        self.point_ = np.empty([0, 3])

    def fit(self):
        """
        Flat Plane Search Algorithm fit
        :return: None
        """

        point_cloud = self.data_.copy()

        while len(point_cloud) >= 0.3 * len(self.data_):
            equation, inliers = self.__plane_detection(in_pc=point_cloud)
            slope = self.__zone_slope(in_eq=equation)

            cloud = np.array([point_cloud[i] for i in inliers])

            clusters = self.__zone_clustering(in_pc=cloud)
            for cluster in clusters:
                area = self.__zone_area(in_pc=cluster)

                print(f">> Equation: {equation}; Slope: {slope}; Area: {area}.")

                # AREA_min = 0.126 m^2
                # ANGLE_max = 15.0 grad
                if (slope <= 15.0 and area >= 0.126) and (slope < self.slope_ or area > self.area_):
                    self.inliers_ = cluster
                    self.equation_ = equation
                    self.slope_ = slope
                    self.area_ = area
                    self.point_ = self.__point_determination(in_pc=cluster)
            point_cloud = np.delete(point_cloud, inliers, axis=0)

        return None

    def __plane_detection(self, in_pc):
        """
        Plane detection in point cloud with RANSAC algorithm
        :param in_pc: Input point cloud
        :return: Plane equation, Inliers points
        """

        # NEED TO OPTIMIZE
        equation, inliers = lib.RANSAC_plane(
            cloud=in_pc,
            thresh=self.scale_ * 2.0,
            max_iter=int(len(in_pc) * 0.01)
        )

        if equation[2] < 0:
            equation = np.array([equation[1], equation[0], -equation[2], -equation[3]])

        return equation, inliers

    def __zone_slope(self, in_eq):
        """
        Zone slope calculation
        :param in_eq: Input equation of plane
        :return: Zone slope
        """

        # rad_phi = (a, b) / (magnitude(a) * magnitude(b))
        # grad_phi = abs(arccos(rad_phi) * 180 / PI)
        normal = np.array([0.0, 0.0, 1.0])
        slope = np.abs(
            np.arccos(
                np.dot(normal, in_eq[:3]) /
                (np.linalg.norm(normal) * np.linalg.norm(in_eq[:3]))
            ) * 180.0 / np.pi
        )

        return slope

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
            algorithm='kd_tree',
            leaf_size=30
        )
        model.fit(X=in_pc)

        num_clusters = set(model.labels_)
        if -1 in num_clusters:
            num_clusters.remove(-1)

        clusters = []
        for num in num_clusters:
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

    def __point_determination(self, in_pc):
        """
        Landing point determination with geometric median
        :param in_pc: Input point cloud
        :return: Point coordinates
        """

        g_median = np.array(lib.geometric_median(in_pc))

        return g_median


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    cloud_shape = np.array([640, 480])
    cloud_step = 0.01

    gen = PC_gen(shape=cloud_shape, step=cloud_step)
    cloud = gen.plane_gen(hiegh=0.5, noise=0.001, loss=0.0)

    time_start = time.time()
    alg = FPS_alg(in_data=cloud, in_scale=cloud_step)
    alg.fit()
    stop_time = time.time() - time_start

    print(f"FPS algorithm result:\n\t- Point: {alg.point_}\n\t- Slope: {alg.slope_}\n\t- Area: {alg.area_}")
    print(f"Time: {stop_time}")
