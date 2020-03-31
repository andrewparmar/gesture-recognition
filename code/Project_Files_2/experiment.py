from utils import video_to_image_array


def run_video_player():

    filename = "person01_walking_d1_uncomp.avi"
    fps = 25

    video_to_image_array(filename, fps)


if __name__ == "__main__":

    run_video_player()
