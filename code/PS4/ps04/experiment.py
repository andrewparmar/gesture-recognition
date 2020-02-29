"""Problem Set 4: Motion Detection"""

import cv2
import os
import numpy as np

import ps4

# I/O directories
input_dir = "input_images"
output_dir = "./"


# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0)):
    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in range(0, v.shape[0], stride):

        for x in range(0, u.shape[1], stride):
            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out


# Functions you need to complete:

def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked
    to select a level in the gaussian pyramid which contains images
    that are smaller than the one located in pyr[0]. This function
    should take the U and V arrays computed from this lower level and
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations
    based on the pyramid level used to obtain both U and V. Multiply
    the result of expand_image by 2 to scale the vector values. After
    each expand_image operation you should adjust the resulting arrays
    to match the current level shape
    i.e. U.shape == pyr[current_level].shape and
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to
                             pyr[0].shape
    """
    # print(f'**** Starting Shape:{u.shape}')
    # print(f'**** Final Goal Shape:{pyr[0].shape}')
    for i in range(level, 0, -1):
        tmp_shape = u.shape
        u_tmp = ps4.expand_image(u)
        v_tmp = ps4.expand_image(v)

        u_tmp *= 2
        v_tmp *= 2

        next_level = i - 1
        next_image = pyr[next_level]
        h, w = next_image.shape
        u = u_tmp[:h, :w]
        v = v_tmp[:h, :w]
    #     print(f'Level:{i}; Before {tmp_shape};\tGoal h:{h},w:{w},\tAfter u:{u.shape}, v:{v.shape}')
    # print(f'**** Final Shape:{u.shape}')

    return u, v


def part_1a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                       'ShiftR2.png'), 0) / 255.
    shift_r5_u5 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                          'ShiftR5U5.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 51
    k_type = "gaussian"
    sigma = 30

    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma,
                             gauss_k_size=5, gauss_sigma_x=10, gauss_sigma_y=1)

    # Flow image
    u_v = quiver(u, v, scale=4, stride=9)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.

    k_size = 49
    k_type = "gaussian"
    sigma = 31
    # smooth input images
    u, v = ps4.optic_flow_lk(shift_0, shift_r5_u5, k_size, k_type, sigma,
                             gauss_k_size=51, gauss_sigma_x=10, gauss_sigma_y=10)

    # Flow image
    u_v = quiver(u, v, scale=1, stride=9)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    images = [shift_r10, shift_r20, shift_r40]
    file_names = ['ps4-1-b-1.png', 'ps4-1-b-2.png', 'ps4-1-b-3.png']

    k_size = 49  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 30  # TODO: Select a sigma value if you are using a gaussian kernel

    for image, file_name in zip(images, file_names):
        u, v = ps4.optic_flow_lk(shift_0, image, k_size, k_type, sigma,
                                 gauss_k_size=49, gauss_sigma_x=22, gauss_sigma_y=1)
        u_v = quiver(u, v, scale=1, stride=10)
        cv2.imwrite(os.path.join(output_dir, file_name), u_v)


def part_2():
    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1',
                                         'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
                yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)

    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
                yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 2  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)

    level_id = 1
    k_size = 51
    k_type = "gaussian"
    sigma = 30
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id], yos_img_02_g_pyr[level_id],
                             k_size, k_type, sigma,
                             gauss_k_size=51, gauss_sigma_x=22, gauss_sigma_y=1)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    levels = 5  # Define the number of pyramid levels
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)

    level_id = 1
    k_size = 51
    k_type = "gaussian"
    sigma = 30
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id], yos_img_03_g_pyr[level_id],
                             k_size, k_type, sigma,
                             gauss_k_size=51, gauss_sigma_x=22, gauss_sigma_y=1)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC
    border_mode = cv2.BORDER_REFLECT101
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    levels = 2
    k_size = 51
    k_type = "gaussian"
    sigma = 11
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=1.5, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

    # You may want to try different parameters for the remaining function
    # calls.
    levels = 2
    k_size = 49
    k_type = "gaussian"
    sigma = 9
    u20, v20 = ps4.hierarchical_lk(shift_0, shift_r20, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u20, v20, scale=1.5, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)

    levels = 3
    k_size = 29
    k_type = "gaussian"
    sigma = 28
    u40, v40 = ps4.hierarchical_lk(shift_0, shift_r40, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u40, v40, scale=1.5, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


def part_4b():
    urban_img_01 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban01.png'), 0) / 255.
    urban_img_02 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban02.png'), 0) / 255.

    levels = 4
    k_size = 51
    k_type = "gaussian"
    sigma = 30
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=1, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))


def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.

    levels, k_size, k_type, sigma = 2, 41, 'gaussian', 21
    interpolation = cv2.INTER_CUBIC
    border_mode = cv2.BORDER_REFLECT101
    u, v = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size,
                               k_type, sigma, interpolation, border_mode,
                               gauss_k_size=k_size, gauss_sigma_x=22,
                               gauss_sigma_y=1)

    frames = [shift_0]

    for t in [0.2, 0.4, 0.6, 0.8]:
        frame_tmp = ps4.warp(shift_r10, (1 - t) * u, (1 - t) * v, interpolation,
                             border_mode)

        frames.append(frame_tmp)

    frames.append(shift_r10)

    # for i, frame in enumerate(frames):
    #     cv2.imwrite(f'part5_new_img_{i}.png', ps4.normalize_and_scale(frame))

    h, w = shift_0.shape
    output_img = np.zeros((2 * h, 3 * w))

    counter = 0

    for j in range(2):
        for i in range(3):
            row = j * h
            col = i * w
            output_img[row:row + h, col:col + w] = frames[counter]
            counter += 1
    # import pdb; [pdb.set_trace()]
    cv2.imwrite('ps4-5-a-1.png', ps4.normalize_and_scale(output_img))


def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    mc_01 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                    'mc01.png'), 0) / 255.
    mc_02 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                    'mc02.png'), 0) / 255.
    mc_03 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                    'mc03.png'), 0) / 255.

    images = [mc_01, mc_02, mc_03]

    levels, k_size, k_type, sigma = 2, 95, 'gaussian', 41
    interpolation = cv2.INTER_CUBIC
    border_mode = cv2.BORDER_REFLECT101

    def _frame_interpolation(img_a, img_b, u, v, t_list):
        I_0 = img_a
        I_1 = img_b
        I_1_warped = ps4.warp(I_1, u, v, interpolation, border_mode)

        frames = [I_0]

        for t in t_list:
            I_prime = (1 - t) * I_0 + t * I_1_warped
            I_t = ps4.warp(I_prime, -t * u, -t * v, interpolation, border_mode)

            frames.append(I_t)

        frames.append(I_1)

        # for l, frame in enumerate(frames):
        #     cv2.imwrite(f'part5b_{l}.png', ps4.normalize_and_scale(frame))

        h, w = img_a.shape
        output_img = np.zeros((2 * h, 3 * w))

        counter = 0

        for j in range(2):
            for i in range(3):
                row = j * h
                col = i * w
                output_img[row:row + h, col:col + w] = frames[counter]
                counter += 1

        return output_img

    for i in range(len(images) - 1):
        u, v = ps4.hierarchical_lk(images[i], images[i + 1], levels, k_size,
                                   k_type, sigma, interpolation, border_mode,
                                   gauss_k_size=k_size, gauss_sigma_x=22, gauss_sigma_y=1)

        # u_v = quiver(u, v, scale=0.9, stride=10)
        # cv2.imshow(f'part 5 quiver {i}', u_v)
        # cv2.waitKey(0)

        output_img = _frame_interpolation(images[i], images[i + 1], u, v,
                                          [0.2, 0.4, 0.6, 0.8])

        cv2.imwrite(f'ps4-5-b-{i+1}.png', ps4.normalize_and_scale(output_img))


def part_6():
    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    raise NotImplementedError


if __name__ == '__main__':
    part_1a()
    part_1b()
    part_2()
    part_3a_1()
    part_3a_2()
    part_4a()
    part_4b()
    part_5a()
    part_5b()
    # part_6()
