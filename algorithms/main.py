import warnings
import time

import numpy as np
import pandas as pd

from FPS_Algorithm import FPS_alg
from CDS_Algorithm import CDS_alg
from SoHS_Algorithm import SoHS_alg

from tests.PC_Generation import PC_gen


tests = np.array([
    [np.array([480, 270]), 0.03333, 0.000, 0.0],
    [np.array([480, 270]), 0.03333, 0.001, 0.0],
    [np.array([480, 270]), 0.03333, 0.002, 0.0],
    [np.array([640, 480]), 0.00625, 0.000, 10.0],
    [np.array([640, 480]), 0.00625, 0.001, 10.0],
    [np.array([640, 480]), 0.00625, 0.002, 10.0],
    [np.array([1280, 720]), 0.01250, 0.000, 20.0],
    [np.array([1280, 720]), 0.01250, 0.001, 20.0],
    [np.array([1280, 720]), 0.01250, 0.002, 20.0],
])


def main():
    test_num = 1
    tests_df = pd.DataFrame([])

    test_num, tests_df = plane_test(test_num, tests_df)
    test_num, tests_df = square_test(test_num, tests_df)
    tests_df.to_excel("tests.xlsx")
    return
    test_num, tests_df = notch_test(test_num, tests_df)

    tests_df.to_excel("tests.xlsx")

    print(f"All {test_num-1} test done!")

    return 0

def plane_test(test_num, tests_df):
    print("PLANE TEMPLATE")
    for test in tests:
        print(test_num)
        cloud_shape, cloud_step, noise, slope = test
        gen = PC_gen(shape=cloud_shape, step=cloud_step)
        cloud = gen.plane_gen(hiegh=1.0, noise=noise, slope=slope)

        FPS = FPS_alg(in_data=cloud, in_scale=cloud_step)
        CDS = CDS_alg(in_data=cloud, in_scale=cloud_step)
        SoHS = SoHS_alg(in_data=cloud, in_shape=cloud_shape, in_scale=cloud_step)

        time_start = time.time()
        FPS.fit()
        stop_time = time.time() - time_start
        tests_df = tests_df.append(pd.DataFrame([[test_num, FPS.area_, FPS.slope_, FPS.deviation_, FPS.point_, stop_time]]), ignore_index=True)

        test_num += 1

    return test_num, tests_df


def square_test(test_num, tests_df):
    print("SQUARE TEMPLATE")
    for test in tests:
        print(test_num)
        cloud_shape, cloud_step, noise, slope = test
        gen = PC_gen(shape=cloud_shape, step=cloud_step)
        cloud = gen.square_gen(h1=0.1, h2=1.0, noise=noise, slope=slope)

        FPS = FPS_alg(in_data=cloud, in_scale=cloud_step)
        CDS = CDS_alg(in_data=cloud, in_scale=cloud_step)
        SoHS = SoHS_alg(in_data=cloud, in_shape=cloud_shape, in_scale=cloud_step)

        time_start = time.time()
        FPS.fit()
        stop_time = time.time() - time_start
        tests_df = tests_df.append(pd.DataFrame([[test_num, FPS.area_, FPS.slope_, FPS.deviation_, FPS.point_, stop_time]]), ignore_index=True)
        return test_num, tests_df
        test_num += 1

    return test_num, tests_df


def notch_test(test_num, tests_df):
    print("NOTCH TEMPLATE")
    for test in tests:
        print(test_num)
        cloud_shape, cloud_step, noise, slope = test
        gen = PC_gen(shape=cloud_shape, step=cloud_step)
        cloud = gen.notch_gen(h1=0.1, h2=1.0, noise=noise, slope=slope)

        FPS = FPS_alg(in_data=cloud, in_scale=cloud_step)
        CDS = CDS_alg(in_data=cloud, in_scale=cloud_step)
        SoHS = SoHS_alg(in_data=cloud, in_shape=cloud_shape, in_scale=cloud_step)

        time_start = time.time()
        FPS.fit()
        stop_time = time.time() - time_start
        tests_df = tests_df.append(pd.DataFrame([[test_num, FPS.area_, FPS.slope_, FPS.deviation_, FPS.point_, stop_time]]), ignore_index=True)

        test_num += 1

    return test_num, tests_df


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()