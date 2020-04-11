import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from config import actions, backgrounds, frame_sequences

matplotlib.use("Qt5Agg")




VID_DIR = "sample_dataset"
WAIT_DURATION = 10
NUM_HU = 7 * 2


class BinaryMotion:
    def __init__(self, len_img_history, threshold):
        self.n = len_img_history
        self.theta = threshold
        self.images = None
        self.last_image = None
        self.binary_image = None
        self.ksize = 2

    def update(self, image):
        if self.images is None:
            h, w = image.shape
            self.images = np.zeros((h, w, self.n), dtype=np.uint8)
            self.images[:, :, -1] = image
            self.last_image = image
        else:
            self.images[:, :, : self.n - 1] = self.images[:, :, 1 : self.n]
            self.images[:, :, -1] = self.last_image
            self.last_image = image

    def get_binary_image(self, mode="frame"):
        """
        binary_image is of type np.float64
        """
        if mode == "median":
            median_image = np.median(self.images[:, :, : self.n - 1], axis=2).astype(
                np.uint8
            )
            diff_image = cv2.absdiff(self.last_image, median_image)
        if mode == "mean":
            median_image = np.mean(self.images[:, :, : self.n - 1], axis=2).astype(
                np.uint8
            )
            diff_image = cv2.absdiff(self.last_image, median_image)
        else:
            diff_image = cv2.absdiff(self.last_image, self.images[:, :, self.n - 1])

        self.binary_image = np.zeros(diff_image.shape)
        self.binary_image[diff_image >= self.theta] = 1
        self._cleanup()

        return self.binary_image

    def _cleanup(self):
        kernel = np.ones((self.ksize, self.ksize), np.uint8)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, kernel)

    def view(self):
        cv2.namedWindow("binary_image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("binary_image", (600, 600))
        cv2.imshow("binary_image", self.binary_image)
        cv2.waitKey(WAIT_DURATION)

    def view_image(self, text=None):
        img = np.copy(self.last_image)
        if text:
            img = self._add_text(img, str(text), (100, 100))

        cv2.namedWindow("last_image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("last_image", (600, 600))
        cv2.imshow("last_image", img)
        cv2.waitKey(WAIT_DURATION)

    @staticmethod
    def _add_text(img, text, coordinate):
        h, w = img.shape
        x, y = coordinate
        font = cv2.FONT_ITALIC
        fontScale = 0.7
        color_outline = (255, 255, 255)
        color_text = (0, 0, 0)
        thickness_outline = 2
        thickness_text = 1

        # text_width, text_height = \
        # cv2.getTextSize(text, font, fontScale, thickness_outline)[0]
        # org = (x - int(text_width / 2), y + 30)
        # while (org[0] + text_width) > w:
        #     org = (org[0] - 1, org[1])
        org = (x, y)

        cv2.putText(img, text, org, font, fontScale, color_outline, thickness_outline)

        return img


class TemporalTemplate:
    def __init__(self, tau):
        self.tau = tau
        self.motion_images = None
        self._mhi = None
        self._mei = None
        # Note: self.mei could just be mhi as float right?

    def update(self, motion_img):
        if self.motion_images is None:
            h, w = motion_img.shape
            self.motion_images = np.zeros((h, w, self.tau), dtype=np.uint8)
            self._mhi = np.zeros((h, w))
        else:
            self.motion_images[:, :, : self.tau - 1] = self.motion_images[
                :, :, 1 : self.tau
            ]

        self.motion_images[:, :, -1] = motion_img

        self._make_motion_history_image()
        self._make_motion_energy_image()

    # def _make_motion_history_image(self):
    #     h, w, n = self.motion_images.shape
    #
    #     mhi = np.zeros((h, w))
    #
    #     tau = self.tau - 1
    #
    #     for i in range(self.tau-1, -1, -1):
    #         # print(f'Tau {tau}, i {i}')
    #         idx = self.motion_images[:, :, i] == 1
    #         mhi[idx] = tau
    #         tau = tau - 1
    #
    #     self.mhi = cv2.normalize(
    #         mhi, -1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    #     )
    # #
    # # # TODO: Also consider, should we be normalizing this? What about making self._mhi and
    # # # exposing mhi as a property?

    def _make_motion_history_image(self):
        h, w, n = self.motion_images.shape

        mhi = np.zeros((h, w))

        idx = self.motion_images[:, :, -1] == 1

        mhi[idx] = self.tau
        mhi[~idx] = self._mhi[~idx] - 1

        mhi[mhi < 0] = 0

        self._mhi = mhi

    @property
    def mhi(self):
        return cv2.normalize(
            self._mhi, -1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

    @property
    def mei(self):
        return cv2.normalize(
            self._mei, -1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

    def _make_motion_energy_image(self):
        mei = np.zeros(self.mhi.shape)
        mei[self._mhi > 0] = 1

        self._mei = mei

    def view(self, type):
        if type == "mhi":
            cv2.namedWindow("motion_history_image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("motion_history_image", (600, 600))
            cv2.imshow("motion_history_image", self.mhi)
            cv2.waitKey(WAIT_DURATION)
        elif type == "mei":
            cv2.namedWindow("motion_energy_image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("motion_energy_image", (600, 600))
            cv2.imshow("motion_energy_image", self.mei)
            cv2.waitKey(WAIT_DURATION)


class HuMoments:
    def __init__(self, image):
        self._moments = self._calc_moments(image)
        self.values = self._calc_hu_moments()

    def _calc_moments(self, image):
        y, x = np.mgrid[: image.shape[0], : image.shape[1]]

        if image.sum() == 0:
            keys = ["nu11", "nu12", "nu21", "nu02", "nu20", "nu03", "nu30"]
            moments = {k: 0 for k in keys}
            return moments
        else:
            x_mean = (x * image).sum() / image.sum()
            y_mean = (y * image).sum() / image.sum()

        moments = {}

        # regular moments
        moments["m00"] = (x ** 0 * y ** 0 * image).sum()
        moments["m10"] = (x ** 1 * y ** 0 * image).sum()
        moments["m01"] = (x ** 0 * y ** 1 * image).sum()
        moments["m11"] = (x ** 1 * y ** 1 * image).sum()
        moments["m20"] = (x ** 2 * y ** 0 * image).sum()
        moments["m02"] = (x ** 0 * y ** 2 * image).sum()
        moments["m30"] = (x ** 3 * y ** 0 * image).sum()
        moments["m03"] = (x ** 0 * y ** 3 * image).sum()
        moments["m12"] = (x ** 1 * y ** 2 * image).sum()
        moments["m21"] = (x ** 2 * y ** 1 * image).sum()

        # central moments
        moments["mu10"] = ((x - x_mean) ** 1 * (y - y_mean) ** 0 * image).sum()
        moments["mu01"] = ((x - x_mean) ** 0 * (y - y_mean) ** 1 * image).sum()
        moments["mu11"] = ((x - x_mean) ** 1 * (y - y_mean) ** 1 * image).sum()
        moments["mu20"] = ((x - x_mean) ** 2 * (y - y_mean) ** 0 * image).sum()
        moments["mu02"] = ((x - x_mean) ** 0 * (y - y_mean) ** 2 * image).sum()
        moments["mu30"] = ((x - x_mean) ** 3 * (y - y_mean) ** 0 * image).sum()
        moments["mu03"] = ((x - x_mean) ** 0 * (y - y_mean) ** 3 * image).sum()
        moments["mu12"] = ((x - x_mean) ** 1 * (y - y_mean) ** 2 * image).sum()
        moments["mu21"] = ((x - x_mean) ** 2 * (y - y_mean) ** 1 * image).sum()

        # scale invariant moments
        moments["nu11"] = moments["mu11"] / moments["m00"] ** (1 + (1 + 1) / 2)
        moments["nu12"] = moments["mu12"] / moments["m00"] ** (1 + (1 + 2) / 2)
        moments["nu21"] = moments["mu21"] / moments["m00"] ** (1 + (2 + 1) / 2)
        moments["nu02"] = moments["mu02"] / moments["m00"] ** (1 + (0 + 2) / 2)
        moments["nu20"] = moments["mu20"] / moments["m00"] ** (1 + (2 + 0) / 2)
        moments["nu03"] = moments["mu03"] / moments["m00"] ** (1 + (0 + 3) / 2)
        moments["nu30"] = moments["mu30"] / moments["m00"] ** (1 + (3 + 0) / 2)

        return moments

    def _calc_hu_moments(self):
        nu11 = self._moments["nu11"]
        nu20 = self._moments["nu20"]
        nu02 = self._moments["nu02"]
        nu30 = self._moments["nu30"]
        nu03 = self._moments["nu03"]
        nu12 = self._moments["nu12"]
        nu21 = self._moments["nu21"]

        h1 = nu20 + nu02  # noqa
        h2 = (nu20 - nu02) ** 2 + 4 * nu11 ** 2  # noqa
        h3 = (nu30 - 3 * nu12) ** 2 + (3 * nu21 - nu03) ** 2  # noqa
        h4 = (nu30 + nu12) ** 2 + (nu21 + nu03) ** 2  # noqa
        h5 = (nu30 - 3 * nu12) * (nu30 + nu12) * (
            (nu30 + nu12) ** 2 - 3 * (nu21 + nu03) ** 2
        ) + (3 * nu21 - nu03) * (nu21 + nu03) * (
            3 * (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2
        )  # noqa
        h6 = (nu20 - nu02) * ((nu30 + nu12) ** 2 - (nu21 + nu03) ** 2) + 4 * nu11 * (
            nu30 + nu12
        ) * (
            nu21 + nu03
        )  # noqa
        h7 = (3 * nu21 - nu03) * (nu30 + nu12) * (
            (nu30 + nu12) ** 2 - 3 * (nu21 + nu03) ** 2
        ) - (nu30 - 3 * nu12) * (nu21 + nu03) * (
            3 * (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2
        )  # noqa

        # hu_moments = np.nan_to_num(np.array([h1, h2, h3, h4, h5, h6, h7]))
        hu_moments = np.array([h1, h2, h3, h4, h5, h6, h7])

        return hu_moments


class ActionVideo:
    TAU_MAX = 40
    TAU_MIN = 10
    TAU = 20
    PARAM_MAP = {
        "boxing": {"label": 1, "theta": 10, "ksize": 2, "tau": TAU},  # noqa
        "handclapping": {"label": 2, "theta": 20, "ksize": 2, "tau": TAU},  # noqa
        "handwaving": {"label": 3, "theta": 35, "ksize": 2, "tau": TAU_MAX},  # noqa
        "jogging": {"label": 4, "theta": 35, "ksize": 2, "tau": TAU},  # noqa
        "running": {"label": 5, "theta": 35, "ksize": 2, "tau": TAU_MIN},  # noqa
        "walking": {"label": 6, "theta": 35, "ksize": 2, "tau": TAU},  # noqa
    }

    def __init__(self, num, action, background):
        self.key_name = f"person{num:02d}_{action}_{background}"
        self.filename = f"{self.key_name}_uncomp.avi"
        self.action = action
        self.frame_ranges = frame_sequences[self.key_name]
        self.video_frame_array = None

        self._video_to_image_array()

    def _video_to_image_array(self):
        input_video_path = os.path.join(VID_DIR, self.filename)

        self.total_video_frames = self._get_video_frame_count(input_video_path)

        input_image_gen = self._gray_frame_generator(input_video_path)
        input_image = input_image_gen.__next__()

        self.video_frame_array = np.zeros(
            (input_image.shape[0], input_image.shape[1], self.total_video_frames),
            dtype=np.uint8,
        )

        frame_num = 0

        while input_image is not None:
            self.video_frame_array[:, :, frame_num] = input_image

            input_image = input_image_gen.__next__()

            frame_num += 1

    def _get_video_frame_count(self, file_path):
        video = cv2.VideoCapture(file_path)

        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        video.release()

        return total

    @staticmethod
    def _gray_frame_generator(file_path):
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
            else:
                break

        video.release()
        yield None

    def analyze_frames(self):
        """
        Uses vstack to accumulate frame arrays.
        """
        binary_image_history = 2
        theta = self.PARAM_MAP[self.action]["theta"]
        tau = self.PARAM_MAP[self.action]["tau"]

        self.frame_features = np.zeros((1, NUM_HU))
        self.frame_labels = np.array([0])

        for frame_range in self.frame_ranges:

            binary_motion = BinaryMotion(binary_image_history, theta)
            temporal_template = TemporalTemplate(tau)

            start = frame_range[0] - 1
            end = frame_range[1]
            # print(f"{start}:{end}")

            for i in range(start, end):
                input_image_t = self.video_frame_array[:, :, i]

                binary_motion.update(input_image_t)
                temporal_template.update(binary_motion.get_binary_image())
                # cv2.imwrite(f"output_images/{i}_binary.jpg", binary_motion.get_binary_image() * 255)
                # cv2.imwrite(f"output_images/{i}_mhi.jpg", temporal_template.mhi)

                # binary_motion.view()
                # binary_motion.view_image(text=i)
                temporal_template.view(type="mhi")
                # temporal_template.view(type="mei")

                # if self.frame_labels.shape[0] == 141:
                #     cv2.imwrite(f"output_images/{counter}_binary.jpg", binary_motion.get_binary_image() * 255)
                #     cv2.imwrite(f"output_images/{counter}_mhi.jpg", temporal_template.mhi)

                hu_moments_mei = HuMoments(temporal_template._mei)
                hu_moments_mhi = HuMoments(temporal_template._mhi)
                hu_moments = np.concatenate(
                    (hu_moments_mei.values, hu_moments_mhi.values)
                )

                # print(hu_moments.values)
                label = np.array([self.PARAM_MAP[self.action]["label"]])

                # print(hu_moments)
                if np.any(hu_moments):
                    self.frame_features = np.vstack((self.frame_features, hu_moments))
                    self.frame_labels = np.vstack((self.frame_labels, label))
                else:
                    self.frame_features = np.vstack((self.frame_features, hu_moments))
                    self.frame_labels = np.vstack((self.frame_labels, 0))

        self.frame_features = self.frame_features[1:, :]
        self.frame_labels = self.frame_labels[1:]

    def analyze_frames_all(self):
        binary_image_history = 2
        theta = self.PARAM_MAP[self.action]["theta"]
        tau = self.PARAM_MAP[self.action]["tau"]

        self.frame_features = np.zeros((self.total_video_frames, self.NUM_HU))
        self.frame_labels = np.zeros(self.total_video_frames)

        for frame_range in self.frame_ranges:

            binary_motion = BinaryMotion(binary_image_history, theta)
            temporal_template = TemporalTemplate(tau)

            start = frame_range[0] - 1
            end = frame_range[1]

            for i in range(start, end):
                input_image_t = self.video_frame_array[:, :, i]

                binary_motion.update(input_image_t)
                temporal_template.update(binary_motion.get_binary_image())

                # binary_motion.view()
                # temporal_template.view(type="mhi")
                hu_moments = HuMoments(temporal_template.mhi)
                print(f"HuMoments: {hu_moments.values}")
                self.frame_features[i] = hu_moments.values


def generate_data(sequence):

    Xtrain = np.zeros((1, NUM_HU))
    ytrain = np.zeros(1)

    for person_num in sequence:
        for action in actions:
            for background in backgrounds[:1]:

                action_video = ActionVideo(person_num, action, background)
                print(action_video.key_name)
                action_video.analyze_frames()

                features = np.log(np.abs(action_video.frame_features))
                features[~np.isfinite(features).any(axis=1)] = np.zeros(NUM_HU)

                # plot_features(features, title=action_video.key_name)

                Xtrain = np.vstack((Xtrain, action_video.frame_features))
                ytrain = np.hstack((ytrain, action_video.frame_labels.reshape(-1)))

    return Xtrain[1:], ytrain[1:]


def plot_features(features, **kwargs):
    labels = ["h1", "h2", "h3", "h4", "h5", "h6", "h7"]

    fig, ax = plt.subplots()

    ax.plot(features[:, :7])
    ax.legend(labels, bbox_to_anchor=(1, 1))
    ax.set(
        xlabel="Frame number", ylabel="Hu Value", title=f'{kwargs.get("title", "Plot")}'
    )

    plt.show()
