import cv2
import pprint
import config

from core import video_frame_sequence_analyzer, generate_training_data


def run_video_player():

    filename = "person01_walking_d1_uncomp.avi"
    fps = 25

    video_frame_sequence_analyzer(filename, fps)
    # video_to_image_array(filename, fps)


if __name__ == "__main__":

    # run_video_player()

    # run_moment_calculation()

    generate_training_data()
