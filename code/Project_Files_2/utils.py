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

    return binary_image


def video_to_image_array(filename, fps):

    input_video_path = os.path.join(VID_DIR, filename)
    input_image_gen = video_gray_frame_generator(input_video_path)
    input_image_t = input_image_gen.__next__()

    frame_num = 1

    while input_image_t is not None:

        input_image_t_minus_1 = input_image_t
        input_image_t = input_image_gen.__next__()

        if frame_num % 10 == 0:
            print("Processing fame {}".format(frame_num))

        # if frame_num == 20:
        #     import pdb; pdb.set_trace()

        binary_image = get_binary_image(input_image_t, input_image_t_minus_1, 30)

        cv2.namedWindow("binary_image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("binary_image", (600, 600))
        cv2.imshow("binary_image", binary_image)
        cv2.waitKey(20)

        if input_image_t is not None:
            input_image_t = input_image_gen.__next__()

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
