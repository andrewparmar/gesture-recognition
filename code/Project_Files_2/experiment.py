import cv2

from utils import moment, video_to_image_array


def run_video_player():

    filename = "person01_walking_d1_uncomp.avi"
    fps = 25

    video_to_image_array(filename, fps)


def run_moment_calculation():
    filename = "mhi_frame_200_person01_walking_d1.png"
    test_image = cv2.imread(filename)

    moment(test_image[:, :, 1])


if __name__ == "__main__":

    # run_video_player()

    run_moment_calculation()
