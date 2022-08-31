import warnings

import numpy as np
import pyransac3d as pyrsc


class WAS_alg:
    """
    Weighted Area Search Algorithm for SLP search
    """

    def __init__(self, in_data=np.array([])):
        self.data = in_data

        self.inliers = []
        self.equation = []

        self.slope = 0.0
        self.area = 0.0
        self.score = 0.0
        self.point = []

    def call(self):
        """
        Weighted Area Search algorithm call
        :return: None
        """
        point_cloud = self.data.copy()

        while len(point_cloud) >= 0.3 * len(self.data):
            equation, inliers = self.plane_detection(in_pc=point_cloud)
            slope = self.zone_slope(in_eq=equation)

            clusters = self.zone_clustering(in_pc=inliers)
            for cluster in clusters:
                area = self.zone_area(in_pc=cluster)
                score = self.zone_estimate(in_area=area, in_slope=slope)
                if score > self.score:
                    self.point = self.point_determination(in_pc=cluster)
            point_cloud = np.delete(point_cloud, inliers, axis=0)
        return None

    def plane_detection(self, in_pc):
        """
        Plane detection in point cloud with RANSAC algorithm
        :param in_pc: Input point cloud
        :return: Plane equation, Inliers points
        """
        plane = pyrsc.Plane()
        equation, inliers = plane.fit(
            pts=in_pc,
            thresh=0.05,
            minPoints=100,
            maxIteration=1000
        )

        return equation, inliers

    def zone_slope(self, in_eq):
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

    def zone_clustering(self, in_pc):
        """
        Zones clustering in point cloud with Kd-tree and Euclidean clustering algorithms
        :param in_pc: Input point cloud
        :return: Zones points clusters
        """
        return []

    def zone_area(self, in_pc):
        """
        Zone area calculation
        :param in_pc: Input point cloud
        :return: Zone area
        """
        return 0

    def zone_estimate(self, in_area, in_slope):
        """
        Landing zone estimation
        :param in_area: Input zone area
        :param in_slope: Input zone slope
        :return: Estimate
        """
        return 0.0

    def point_determination(self, in_pc):
        """
        Landing point determination with geometric median
        :param in_pc: Input point cloud
        :return: Point coordinates
        """
        return 0.0, 0.0, 0.0


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    input = np.array([np.array([i, j, 0.5]) for j in range(100) for i in range(100)])

    alg = WAS_alg(in_data=input)
    alg.call()

    print(f"PC characteristic:\n\t- Point: {alg.point}\n\t- Equation: {alg.equation}\n\t- Area: {alg.area}")