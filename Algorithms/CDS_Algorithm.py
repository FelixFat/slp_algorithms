import numpy as np


class CDS_alg:
    """
    Cluster Dispersion Search Algorithm
    """

    def __init__(self, in_data=np.empty([0, 3])):
        self.data = in_data.copy()

        self.inliers_ = np.array([])
        self.equation_ = np.array([])

        self.scale_ = 0.01
        self.score_ = 0.0

        self.slope_ = 0.0
        self.area_ = 0.0
        self.scatter_ = 0.0
        self.point_ = np.array([])

    def fit(self):
        return None

    def __zone_clustering(self):
        return np.empty([0, 3])

    def __zone_estimate(self):
        return 0.0

    def __point_determination(self):
        return np.array([0.0, 0.0, 0.0])


if __name__ == '__main__':
    alg = CDS_alg()
    alg.fit()