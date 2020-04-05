import cv2
import pprint

from utils import moments, video_to_image_array, hu_moments


def run_video_player():

    filename = "person01_walking_d1_uncomp.avi"
    fps = 25

    video_to_image_array(filename, fps)


def run_moment_calculation():
    filename = "mhi_frame_200_person01_walking_d1.png"
    test_image = cv2.imread(filename)

    moments_ = moments(test_image[:, :, 0])
    # pprint.pprint(moments_)

    pprint.pprint(hu_moments(moments_))

    cv2_moments = cv2.moments(test_image[:, :, 0])

    cv2.HuMoments(cv2_moments)

if __name__ == "__main__":

    run_video_player()

    # run_moment_calculation()
