import pprint
import numpy as np
import cv2

import config
import core

def run_moment_calculation():
    filename = "mhi_frame_200_person01_walking_d1.png"
    test_image = cv2.imread(filename)

    for type_ in [np.uint8, np.uint16, np.float]:
        print(f'\nType" {type_}')
        test_image = test_image.astype(type_)

        cat = core.HuMoments(test_image[:, :, 0])
        pprint.pprint(cat.values)

        cv2_moments = cv2.moments(test_image[:, :, 0])
        pprint.pprint(cv2.HuMoments(cv2_moments).flatten())

if __name__ == "__main__":
    np.set_printoptions(precision=5, linewidth=200)

    run_moment_calculation()

    Xtrain, ytrain = core.generate_training_data()
