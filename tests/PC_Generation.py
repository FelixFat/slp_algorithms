import numpy as np


class PC_gen:
    """
    Point clouds generation class
    """

    def __init__(self, shape=np.array([100, 100]), step=0.01):
        self.shape_ = shape.copy()
        self.step_ = step

        self.row_, self.col_ = (self.shape_ / 2) * self.step_

    def __point_loss(self, quant, loss):
        """
        Lost point indexes generator
        :param quant: Point quantity in the cloud
        :param loss: Point loss percentage
        :return: Lost point indexes
        """

        size = int(np.floor(quant * loss))
        cloud_id = np.random.randint(low=0, high=quant, size=size, dtype=np.uint32)

        return cloud_id

    def plane_gen(self, hiegh=0.5, noise=0.001, loss=0.0):
        """
        Plane point cloud generator
        :param noise: Point cloud noise
        :param loss: Point loss percentage
        :return: Point cloud
        """

        cloud = np.array([
            np.array([i, j, hiegh])
            for j in np.arange(-self.col_, self.col_, self.step_)
            for i in np.arange(-self.row_, self.row_, self.step_)
        ])
        cloud += np.random.normal(0, noise, cloud.shape)
        cloud = np.delete(cloud, self.__point_loss(quant=len(cloud), loss=loss), axis=0)

        return cloud
