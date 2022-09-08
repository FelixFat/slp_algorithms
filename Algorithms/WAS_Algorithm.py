import warnings

import numpy as np
import pyransac3d as pyrsc
from sklearn.cluster import DBSCAN

import library


class WAS_alg:
    """
    Weighted Area Search Algorithm for SLP search
    """

    def __init__(self, in_data=np.array([])):
        self.data_ = in_data

        self.inliers_ = []
        self.equation_ = []

        self.scale_ = 0.01

        self.slope_ = 0.0
        self.area_ = 0.0
        self.score_ = 0.0
        self.point_ = []

    def call(self):
        """
        Weighted Area Search algorithm call
        :return: None
        """

        point_cloud = self.data_.copy()

        while len(point_cloud) >= 0.3 * len(self.data_):
            equation, inliers = self._plane_detection(in_pc=point_cloud)
            slope = self._zone_slope(in_eq=equation)

            pc = np.array([point_cloud[i] for i in inliers])

            clusters = self._zone_clustering(in_pc=pc)
            for cluster in clusters:
                area = self._zone_area(in_pc=cluster)
                score = self._zone_estimate(in_pc=cluster, in_area=area, in_slope=slope)

                print(f">> Equation: {equation}. Parameters: {slope}, {area}, {score}")
                if score > self.score_:
                    self.inliers_ = cluster
                    self.equation_ = equation
                    self.slope_ = slope
                    self.area_ = area
                    self.score_ = score
                    self.point_ = self._point_determination(in_pc=cluster)
            point_cloud = np.delete(point_cloud, inliers, axis=0)

        return None

    def _plane_detection(self, in_pc):
        """
        Plane detection in point cloud with RANSAC algorithm
        :param in_pc: Input point cloud
        :return: Plane equation, Inliers points
        """

        # NEED TO CHECK (dependence on random)
        plane = pyrsc.Plane()
        equation, inliers = plane.fit(
            pts=in_pc,
            thresh=0.05,
            minPoints=100,
            maxIteration=1000
        )

        return equation, inliers

    def _zone_slope(self, in_eq):
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

    def _zone_clustering(self, in_pc):
        """
        Zones clustering in point cloud with Kd-tree and Euclidean clustering algorithms
        :param in_pc: Input point cloud
        :return: Zones points clusters
        """

        model = DBSCAN(
            eps=2,
            min_samples=5,
            metric='euclidean',
            algorithm='kd_tree',
            leaf_size=30
        )
        model.fit(X=in_pc)

        num_clusters = set(model.labels_)
        if -1 in num_clusters:
            num_clusters.remove(-1)

        clusters = []
        # NEED TO OPTIMIZE (calculations)
        for num in num_clusters:
            arr = np.array([
                in_pc[i]
                for i in range(len(model.labels_))
                if model.labels_[i] == num
                ])
            clusters.append(arr)

        return clusters

    def _zone_area(self, in_pc):
        """
        Zone area calculation
        :param in_pc: Input point cloud
        :return: Zone area
        """

        square = self.scale_ ** 2
        area = len(in_pc) * square

        return area

    def _zone_estimate(self, in_pc, in_area, in_slope):
        """
        Landing zone estimation
        :param in_area: Input zone area
        :param in_slope: Input zone slope
        :return: Estimate
        """

        # AREA_min = 0.126 m^2
        # ANGLE_max = 15.0 grad
        est = (in_area / 0.126) * (1.0 - in_slope / 15.0)

        return est

    def _point_determination(self, in_pc):
        """
        Landing point determination with geometric median
        :param in_pc: Input point cloud
        :return: Point coordinates
        """

        # NEED TO CHECK (linearized method)
        g_median = np.array(library.geometric_median(in_pc))

        return g_median


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    input = np.array([np.array([i, j, 0.5]) for j in range(100) for i in range(100)])
    input[2] = [1000, -1000, 0.5]
    input[5] = [-1000, -1001, 0.5]
    input[10] = [-1001, -1000, 0.5]

    alg = WAS_alg(in_data=input)
    alg.call()

    print(f"WAS algorithm result:\n\t- Point: {alg.point_}\n\t- Equation: {alg.equation_}\n\t- Area: {alg.area_}")