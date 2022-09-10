import numpy as np
from sklearn.cluster import DBSCAN

import Int_Library as lib


class CDS_alg:
    """
    Cluster Dispersion Search Algorithm
    """

    def __init__(self, in_data=np.empty([0, 3])):
        self.data_ = in_data.copy()

        self.inliers_ = np.array([])
        self.equation_ = np.array([])

        self.scale_ = 0.01
        self.score_ = 0.0

        self.slope_ = 0.0
        self.area_ = 0.0
        self.scatter_ = 0.0
        self.point_ = np.array([])

    def fit(self):
        point_cloud = self.data_.copy()

        clusters = self.__zone_clustering(in_pc=point_cloud)
        for cluster in clusters:
            equation, slope = self.__zone_slope(in_pc=cluster)
            area = self.__zone_area(in_pc=cluster)
            scatter = self.__zone_scatter(in_pc=cluster, in_eq=equation)

            score = self.__zone_estimate(in_pc=cluster, in_slope=slope, in_area=area, in_scatter=scatter)
            print(f">> Equation: {equation}; Slope: {slope}; Area: {area}; Scatter: {scatter}. Score: {score}.")

            if score > self.score_:
                self.inliers_ = cluster
                self.equation_ = equation
                self.score_ = score
                self.slope_ = slope
                self.area_ = area
                self.scatter_ = scatter
                self.point_ = self.__point_determination(in_pc=cluster)

        return None

    def __zone_clustering(self, in_pc):
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

    def __zone_scatter(self, in_pc, in_eq):
        """
        Zone scatter calculation
        :param in_pc: Input point cloud
        :param in_eq: Input zone equation
        :return: Zone area
        """

        # Отклонение/разброс точек от срелней плоскости
        mse_lamb = lambda A, B: np.power(np.sum(A * B), 2)
        mse = np.sum([mse_lamb(np.append(point, [1.0]), in_eq) for point in in_pc])
        scatter = mse / len(in_pc)

        return scatter

    def __zone_estimate(self, in_pc, in_slope, in_area, in_scatter):
        """
        Zone estimate calculation
        :param in_slope:
        :param in_area:
        :param in_scatter:
        :return:
        """

        area_cl = in_area / self.__zone_area(in_pc=in_pc)
        scatter_cl = in_scatter / len(in_pc)
        slope_cl = in_slope / 15.0

        est = area_cl * (1 - 0.5 * (scatter_cl + slope_cl))

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
    data = np.array([list([i, j, 0.5]) for j in range(100) for i in range(100)])
    data = data + np.random.normal(0, 0.02, data.shape)  # add noise

    alg = CDS_alg(in_data=data)
    alg.fit()

    print(f"CDS algorithm result:\n\t- Point: {alg.point_}\n\t- Slope: {alg.slope_}\n\t- Area: {alg.area_}\n\t- Scatter: {alg.scatter_}")