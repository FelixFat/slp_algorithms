import numpy as np
from scipy.spatial.distance import cdist, euclidean


def RANSAC_plane(cloud, thresh=0.05, max_iter=1000):
    """
    RANSAC plane detection
    (based on https://github.com/leomariga/pyRANSAC-3D/blob/master/pyransac3d/plane.py)
    :param cloud: Input point cloud
    :param thresh: thresh for plane
    :param min_points: Minimum points for detectable plane
    :param max_iter: Maximum iterations for algorithm
    :return: Plane equation, Inliers points
    """

    best_eq, best_inliers = np.empty([0, 4]), np.empty([0, 3])

    np.random.seed(42)
    for _ in range(max_iter):
        samples = np.array([
            cloud[ind]
            for ind in np.random.randint(low=0, high=len(cloud), size=3, dtype=np.uint32)
        ])

        vecA = samples[1, :] - samples[0, :]
        vecB = samples[2, :] - samples[0, :]

        cross_prod = np.cross(vecA, vecB)
        vecABC = np.divide(cross_prod, np.linalg.norm(cross_prod))

        k = -np.sum(np.multiply(vecABC, samples[1, :]))
        plane_eq = np.hstack([vecABC, np.array([k])])

        distances = np.divide(
            (plane_eq[0] * cloud[:, 0] + plane_eq[1] * cloud[:, 1] + plane_eq[2] * cloud[:, 2] + plane_eq[3]),
            np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)
        )

        id_inliers = np.where(np.abs(distances) <= thresh)[0]
        if len(id_inliers) > len(best_inliers):
            best_eq = plane_eq
            best_inliers = id_inliers

    return best_eq, best_inliers


def geometric_median(X, eps=1e-5):
    """
    The multivariate L1 -median and associated data depth
    (based on https://www.pnas.org/doi/pdf/10.1073/pnas.97.4.1423)
    :param X: Input point cloud
    :param eps: Input epsilon
    :return: Geometric median point
    """

    y = np.mean(X, axis=0)
    while True:
        D = cdist(X, np.array([y]))
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)

        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y

        R = (T - y) * Dinvs
        r = np.linalg.norm(R)
        rinv = 0 if r == 0 else num_zeros / r
        y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def divide_and_conquer(shape):
    """
    Divide and Conquer Algorithm
    :param shape: Frame shape
    :return: Base frame shape
    """
    if shape[0] == shape[1] and (shape[0] != 1.0 and shape[1] != 1.0):
        return shape
    elif shape[0] == shape[1] and (shape[0] != 1.0 and shape[1] != 1.0):
        return None

    if shape[0] > shape[1]:
        shape[0] -= shape[1]
    else:
        shape[1] -= shape[0]

    return divide_and_conquer(shape=shape)
