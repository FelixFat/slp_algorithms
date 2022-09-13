import numpy as np
import matplotlib.pyplot as plt


class PC_gen:
    """
    Point clouds generation class
    """

    def __init__(self, shape=np.array([100, 100]), step=0.01):
        self.shape_ = shape.copy()
        self.step_ = step

        self.row_, self.col_ = (self.shape_ / 2) * self.step_

    def visualization(self, cloud):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2])

        ax.set_title('Исходное облако точек')
        ax.set_xlabel('Ось X')
        ax.set_ylabel('Ось Y')
        ax.set_zlabel('Ось Z')
        plt.show()
        return

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

    def plane_gen(self, hiegh=0.5, noise=0.001, slope=0.0, loss=0.0):
        """
        Plane point cloud generator
        :param noise: Point cloud noise
        :param loss: Point loss percentage
        :return: Point cloud
        """

        equation = np.array([0.0, 0.0, 1.0])
        if slope == 10.0:
            equation = np.array([0.0, 0.5, 2.83561])
        elif slope == 20.0:
            equation = np.array([0.0, 0.5, 1.37373])
        z_calc = lambda eq, x, y, h: (eq[0] * x + eq[1] * y + h) / eq[2]

        cloud = np.array([
            np.array([j, i, z_calc(equation, i, j, hiegh)])
            for i in np.arange(-self.col_, self.col_, self.step_)
            for j in np.arange(-self.row_, self.row_, self.step_)
        ])
        cloud += np.random.normal(0, noise, cloud.shape)
        cloud = np.delete(cloud, self.__point_loss(quant=len(cloud), loss=loss), axis=0)

        return cloud


    def square_gen(self, h1=0.1, h2=1.0, noise=0.001, slope=0.0, loss=0.0):

        equation = np.array([0.0, 0.0, 1.0])
        if slope == 10.0:
            equation = np.array([0.0, 0.5, 2.83561])
        elif slope == 20.0:
            equation = np.array([0.0, 0.5, 1.37373])
        z_calc = lambda eq, x, y, h: (eq[0] * x + eq[1] * y + h) / eq[2]

        cloud = np.array([
            np.array([j, i, h1])
            for i in np.arange(-self.col_, self.col_, self.step_)
            for j in np.arange(-self.row_, self.row_, self.step_)
        ])

        for ind in range(len(cloud)):
            if ind // self.shape_[0] in range(40, self.shape_[1] - 40) and \
                ind % self.shape_[0] in range(40, self.shape_[0] - 40):
                    cloud[ind, 2] = z_calc(equation, ind // self.shape_[0], ind % self.shape_[0], h2)

        cloud += np.random.normal(0, noise, cloud.shape)
        cloud = np.delete(cloud, self.__point_loss(quant=len(cloud), loss=loss), axis=0)

        return cloud


    def notch_gen(self, h1=0.1, h2=1.0, noise=0.001, slope=0.0, loss=0.0):

        equation = np.array([0.0, 0.0, 1.0, 1.0])
        if slope == 10.0:
            equation = np.array([0.0, 0.5, 2.83561, -600])
        elif slope == 20.0:
            equation = np.array([0.0, 0.5, 1.37373, -600])
        z_calc = lambda eq, x, y, h: (eq[0] * x + eq[1] * y + eq[3] * h) / eq[2]

        cloud = np.array([
            np.array([j, i, h2])
            for i in np.arange(-self.col_, self.col_, self.step_)
            for j in np.arange(-self.row_, self.row_, self.step_)
        ])

        for ind in range(len(cloud)):
            if ind // self.shape_[0] in range(self.shape_[1] - 100, self.shape_[1]) and \
                ind % self.shape_[0] in range(120, self.shape_[0] - 120):
                    cloud[ind, 2] = z_calc(equation, ind // self.shape_[0], ind % self.shape_[0], h1)

        cloud += np.random.normal(0, noise, cloud.shape)
        cloud = np.delete(cloud, self.__point_loss(quant=len(cloud), loss=loss), axis=0)

        return cloud
