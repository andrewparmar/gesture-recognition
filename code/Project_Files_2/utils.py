import os

import cv2
import numpy as np

VID_DIR = "sample_dataset"

# def helper_for_part_4_and_5(video_name, fps, frame_ids, output_prefix,
#                             counter_init, is_part5):
#
#     video = os.path.join(VID_DIR, video_name)
#     image_gen = ps3.video_gray_frame_generator(video)
#
#     image = image_gen.__next__()
#     h, w, d = image.shape
#
#     out_path = "ar_{}-{}".format(output_prefix[4:], video_name)
#     video_out = mp4_video_writer(out_path, (w, h), fps)
#
#     # Optional template image
#     template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))
#
#     if is_part5:
#         advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
#         src_points = ps3.get_corners_list(advert)
#
#     output_counter = counter_init
#
#     frame_num = 1
#
#     while image is not None:
#
#         print("Processing fame {}".format(frame_num))
#
#         markers = ps3.find_markers(image, template)
#
#         if is_part5:
#             homography = ps3.find_four_point_transform(src_points, markers)
#             image = ps3.project_imageA_onto_imageB(advert, image, homography)
#
#         else:
#
#             for marker in markers:
#                 mark_location(image, marker)
#
#         frame_id = frame_ids[(output_counter - 1) % 3]
#
#         if frame_num == frame_id:
#             out_str = output_prefix + "-{}.jpg".format(output_counter)
#             save_image(out_str, image)
#             output_counter += 1
#
#         video_out.write(image)
#
#         image = image_gen.__next__()
#
#         frame_num += 1
#
#     video_out.release()


def get_binary_image(image1, image2, threshold):
    # TODO: median filtering. See 8D-L1 lecture

    diff_image = cv2.absdiff(image1, image2)

    binary_image = np.zeros(diff_image.shape)

    idx = diff_image >= threshold

    binary_image[idx] = 1

    binary_image = cleanup_image(binary_image)

    return binary_image


# def make_motion_history_image(binary_sequence):
#     # Second try: Recursive reduction
#     h, w, n = binary_sequence.shape
#
#     mhi = np.zeros((h, w))
#
#     tau = 255
#
#     if n == 1:
#         idx = binary_sequence[:, :, 0] == 1
#         mhi[idx] = tau
#         mhi[~idx] = 0
#         return mhi
#     else:
#         idx = binary_sequence[:, :, 0] == 1
#         mhi[idx] = tau
#         mhi[~idx] = np.maximum(make_motion_history_image(binary_sequence[:, :, 1:]) - 30, 0)[~idx]
#         return mhi


def make_motion_history_image(binary_sequence):
    # First try: Reduce images by set weight
    # import pdb; pdb.set_trace()
    h, w, n = binary_sequence.shape

    mhi = np.zeros((h, w))

    tau = 255 / n

    for i in range(n):
        print(f"This is {i}th frame. Tau: {255 - tau*i}")
        idx = binary_sequence[:, :, i] == 1
        mhi[idx] = tau * i

    return mhi


def cleanup_image(image):
    kernel = np.ones((5, 5), np.uint8)
    cleaned_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    return cleaned_image


def video_to_image_array(filename, fps):

    input_video_path = os.path.join(VID_DIR, filename)
    input_image_gen = video_gray_frame_generator(input_video_path)
    input_image_t = input_image_gen.__next__()

    h, w = input_image_t.shape
    n = 50  # 50 is best so far
    tau = 40  # 50 is best so far
    q = 10

    frame_num = 0

    last_n_images = np.zeros((h, w, n), dtype=np.uint8)

    last_q_binary_images = np.zeros((h, w, q), dtype=np.uint8)

    while input_image_t is not None:

        last_n_images[:, :, : n - 1] = last_n_images[:, :, 1:n]
        last_n_images[:, :, -1] = input_image_t

        input_image_t = input_image_gen.__next__()

        if frame_num % 10 == 0:
            print("Processing fame {}".format(frame_num))

        # if frame_num == 200:
        #     import pdb; pdb.set_trace()

        median_of_past_n_images = np.median(last_n_images, axis=2)

        binary_image = get_binary_image(
            input_image_t, median_of_past_n_images.astype(np.uint8), tau
        )
        # binary_image = cleanup_image(binary_image)

        last_q_binary_images[:, :, : q - 1] = last_q_binary_images[:, :, 1:q]
        last_q_binary_images[:, :, -1] = binary_image

        # motion_history_image = make_motion_history_image(np.flip(last_q_binary_images, 2))
        motion_history_image = make_motion_history_image(last_q_binary_images)

        cv2.namedWindow("binary_image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("binary_image", (600, 600))
        cv2.imshow("binary_image", binary_image)

        cv2.namedWindow("motion_history_image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("motion_history_image", (600, 600))
        cv2.imshow("motion_history_image", motion_history_image.astype(np.uint8))

        cv2.waitKey(10)

        # if input_image_t is not None:
        #     input_image_t = input_image_gen.__next__()

        frame_num += 1


def video_gray_frame_generator(file_path):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        file_path (string): Relative file path.

    Returns:
        None.
    """
    video = cv2.VideoCapture(file_path)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield gray_image
            # yield gray_image
        else:
            break

    video.release()
    yield None
