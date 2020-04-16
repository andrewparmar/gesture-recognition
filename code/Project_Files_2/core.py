import os
from collections import deque

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import normalize

import config
import utils
from config import (
    NUM_HU,
    TAU,
    TAU_MAX,
    TAU_MIN,
    THETA,
    VID_DIR,
    WAIT_DURATION,
    frame_sequences,
)

matplotlib.use("Qt5Agg")


class BinaryMotion:
    def __init__(self, n_history, threshold, ksize=2):
        self.n = n_history
        self.theta = threshold
        self.images = None
        self.last_image = None
        self.binary_image = None
        self.ksize = ksize

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
        elif mode == "mean":
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
            img = utils.add_text_to_img(img, str(text), (100, 100))

        cv2.namedWindow("last_image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("last_image", (600, 600))
        cv2.imshow("last_image", img)
        cv2.waitKey(WAIT_DURATION)


class TemporalTemplate:
    def __init__(self, tau):
        self.tau = tau
        self.motion_images = None
        self.mhi = None
        self.mei = None

    def update(self, motion_img):
        if self.motion_images is None:
            h, w = motion_img.shape
            self.motion_images = np.zeros((h, w, self.tau), dtype=np.uint8)
            self.mhi = np.zeros((h, w))
        else:
            self.motion_images[:, :, : self.tau - 1] = self.motion_images[
                :, :, 1 : self.tau
            ]

        self.motion_images[:, :, -1] = motion_img

        self._make_motion_history_image()
        self._make_motion_energy_image()

    def _make_motion_history_image(self):
        h, w, n = self.motion_images.shape

        mhi = np.zeros((h, w))

        idx = self.motion_images[:, :, -1] == 1

        mhi[idx] = self.tau
        mhi[~idx] = self.mhi[~idx] - 1
        mhi[mhi < 0] = 0

        self.mhi = mhi

    def _make_motion_energy_image(self):
        mei = np.zeros(self.mhi.shape)
        mei[self.mhi > 0] = 1

        self.mei = mei

    @property
    def _mhi(self):
        return cv2.normalize(
            self.mhi, -1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

    @property
    def _mei(self):
        return cv2.normalize(
            self.mei, -1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

    def view(self, type):
        if type == "mhi":
            cv2.namedWindow("motion_history_image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("motion_history_image", (600, 600))
            cv2.imshow("motion_history_image", self._mhi)
            cv2.waitKey(WAIT_DURATION)
        elif type == "mei":
            cv2.namedWindow("motion_energy_image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("motion_energy_image", (600, 600))
            cv2.imshow("motion_energy_image", self._mei)
            cv2.waitKey(WAIT_DURATION)


class HuMoments:
    def __init__(self, image):
        self._moments = self._calc_moments(image)
        self.values = self._calc_hu_moments()

    def _calc_moments(self, image):
        y, x = np.mgrid[: image.shape[0], : image.shape[1]]

        if image.sum() == 0:
            keys = ["nu11", "nu12", "nu21", "nu02", "nu20", "nu03", "nu30"]
            moments = {k: 0.0 for k in keys}
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

        hu_moments = np.array([h1, h2, h3, h4, h5, h6, h7])

        return hu_moments


class ActionVideo:

    PARAM_MAP = {
        "undefined": {"label": 0, "theta": 20, "ksize": 2, "tau": TAU},  # noqa,
        "boxing": {"label": 1, "theta": 20, "ksize": 2, "tau": TAU_MIN},  # noqa
        "handclapping": {"label": 2, "theta": 20, "ksize": 2, "tau": TAU},  # noqa
        "handwaving": {"label": 3, "theta": 20, "ksize": 2, "tau": TAU_MAX},  # noqa
        "jogging": {"label": 4, "theta": 20, "ksize": 2, "tau": TAU},  # noqa
        "running": {"label": 5, "theta": 20, "ksize": 2, "tau": TAU_MIN},  # noqa
        "walking": {"label": 6, "theta": 20, "ksize": 2, "tau": TAU_MAX},  # noqa,
    }

    def __init__(self, num, action, background):
        self.key_name = f"person{num:02d}_{action}_{background}"
        self.filename = f"{self.key_name}_uncomp.avi"
        self.action = action
        self.frame_ranges = frame_sequences[self.key_name]
        self.video_frame_array = None

        self._video_to_image_array()

    def __repr__(self):
        return f"<ActionVideo {self.key_name}>"

    def _video_to_image_array(self):
        input_video_path = os.path.join(VID_DIR, self.filename)

        self.total_video_frames = utils.get_video_frame_count(input_video_path)

        input_image_gen = utils.gray_frame_generator(input_video_path)
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

    def _get_range_set(self):
        full_set = set()

        for frame_range in self.frame_ranges:
            a, b = frame_range
            range_set = {x for x in range(a - 1, b)}

            full_set = full_set | range_set

        return full_set

    def analyze_frames(self):
        binary_image_history = 2
        theta = self.PARAM_MAP[self.action]["theta"]
        tau = self.PARAM_MAP[self.action]["tau"]
        ksize = self.PARAM_MAP[self.action]["ksize"]

        self.frame_features = np.zeros((self.total_video_frames, NUM_HU))
        self.frame_labels = np.zeros(self.total_video_frames)

        full_set = self._get_range_set()

        binary_motion = BinaryMotion(binary_image_history, theta, ksize)
        temporal_template = TemporalTemplate(tau)

        for i in range(self.total_video_frames):
            input_image_t = self.video_frame_array[:, :, i]

            binary_motion.update(input_image_t)
            temporal_template.update(binary_motion.get_binary_image())

            # binary_motion.view()
            # binary_motion.view_image(text=i)
            # temporal_template.view(type="mhi")
            # temporal_template.view(type="mei")

            hu_mei = HuMoments(temporal_template.mei)
            hu_mhi = HuMoments(temporal_template.mhi / temporal_template.tau)
            hu_all = np.concatenate((hu_mei.values, hu_mhi.values))
            feature_arr = np.log(np.abs(hu_all))

            if np.any(np.isinf(feature_arr)):
                self.frame_features[i] = np.zeros(feature_arr.shape)
            elif i not in full_set:
                self.frame_features[i] = feature_arr
            else:
                self.frame_features[i] = feature_arr
                self.frame_labels[i] = np.array([self.PARAM_MAP[self.action]["label"]])

            utils.print_fraction(i, self.total_video_frames)

    def plot_features_by_frame(self, ax=None):
        print(f"Generating plot for video {self.key_name}")

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        labels = ["h1", "h2", "h3", "h4", "h5", "h6", "h7"]
        theta = self.PARAM_MAP[self.action]["theta"]
        tau = self.PARAM_MAP[self.action]["tau"]
        title = f"""{self.key_name} \u03C4={tau}; \u03B8={theta}"""
        features = np.abs(self.frame_features[:100, :7])

        lines = ax.plot(features)

        ax.set_ylim([0, features.max() * 1.1])
        ax.set(xlabel="Frame number", ylabel="log(abs(HuValue))")
        ax.set_title(title, fontsize=15)  # title of plo
        ax.legend(lines, labels, loc=(0.005, 0.92), ncol=7, columnspacing=1)
        ax.grid(True)


class ActionVideoUnknownTau(ActionVideo):
    num_windows = TAU_MAX - TAU_MIN + 1

    def __init__(self, classifier, filename, action="undefined", analyze=True):
        self.classifier = classifier
        self.filename = filename
        self.action = action
        self.buffer = deque([], maxlen=15)

        self._video_to_image_array()

        # This lets _get_range_set return all frames ids.
        self.frame_ranges = [(1, self.total_video_frames)]  # TODO: What is this? Why?

        if analyze:
            print("Starting analyze frames")
            self.analyze_frames()

            print("Starting analyze frames backwards tau")
            self.analyze_frame_backwards_tau()

    def get_feature_sequence(self, mhi):
        num_windows = TAU_MAX - TAU_MIN + 1

        feature_sequence = np.zeros((num_windows, NUM_HU))

        for i in range(num_windows):
            mhi_t = mhi - i
            mhi_t[mhi_t < 0] = 0
            mei_t = np.zeros(mhi.shape)
            mei_t[mhi_t > 0] = 1

            if mhi_t.max():
                hu_moments_mhi = HuMoments(mhi_t / mhi_t.max())
            else:
                hu_moments_mhi = HuMoments(mhi_t)
            hu_moments_mei_t = HuMoments(mei_t)
            hu_moments = np.concatenate(
                (hu_moments_mei_t.values, hu_moments_mhi.values)
            )

            feature_arr = np.log(np.abs(hu_moments))

            if np.any(np.isinf(feature_arr)):
                feature_arr = np.zeros(feature_arr.shape)

            feature_sequence[i] = feature_arr

        return feature_sequence

    def frame_feature_set_generator(self):
        """
        We need a generator because we need N feature values for each frame.

        Need a 3d array to store all values and them apply all log values.
        Complicates things a bit.
        """
        binary_image_history = 2
        theta = THETA

        binary_motion = BinaryMotion(binary_image_history, theta)
        temporal_template = TemporalTemplate(TAU_MAX)

        for i in range(self.total_video_frames):
            input_image_t = self.video_frame_array[:, :, i]

            binary_motion.update(input_image_t)
            temporal_template.update(binary_motion.get_binary_image())

            features_sequence = self.get_feature_sequence(temporal_template.mhi)

            yield features_sequence

        raise StopIteration

    def analyze_frame_backwards_tau(self):
        n_features_sequence = np.zeros(
            (self.num_windows, NUM_HU, self.total_video_frames)
        )

        for i, features_set in enumerate(self.frame_feature_set_generator()):

            n_features_sequence[:, :, i] = features_set

            utils.print_fraction(i, self.total_video_frames)

        self.n_features_sequence = n_features_sequence


class VideoActionLabeler(ActionVideoUnknownTau):
    LABELS = {
        0: "no action",
        1: "boxing",
        2: "clapping",
        3: "waving",
        4: "jogging",
        5: "running",
        6: "walking",
    }

    def __init__(self, classifier, filename, fps):
        self.fps = fps
        super(VideoActionLabeler, self).__init__(classifier, filename, analyze=False)

    def create_annotated_video(self):
        h, w, _ = self.video_frame_array.shape

        filename = self.filename.split(".")[0]

        out_path = f"{config.OUTPUT_DIR}/labeled_{filename}.mp4"

        video_out = utils.mp4_video_writer(out_path, (w, h), self.fps)

        for i, feature_set in enumerate(self.frame_feature_set_generator()):

            action_pred, freq_pred = self.classifier._predict_from_feature_set(
                feature_set
            )

            if action_pred == 0:
                label = self.LABELS[0]
            else:
                label = self.LABELS[freq_pred]

            annotated_frame = np.copy(self.video_frame_array[:, :, i])
            annotated_frame = utils.add_text_to_img(annotated_frame, label, (50, 100))
            out_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_GRAY2BGR)

            cv2.namedWindow("annotated_frames", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("annotated_frames", (600, 600))
            cv2.imshow("annotated_frames", out_frame)
            cv2.waitKey(1)

            out_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_GRAY2BGR)

            video_out.write(out_frame)

            utils.print_fraction(i, self.total_video_frames)

        video_out.release()


class ModifiedRandomForest:
    """
    This is pretty much a regular random forest with a wrapper around it to allow us to
    call predict on a backward looking tau feature set.
    """

    def __init__(self, trained_rfc, buffer_len=15, use_action=False):
        self.classifier = trained_rfc
        self._estimator_type = "classifier"
        self.buffer = deque([], maxlen=buffer_len)
        self.use_action = use_action

    def _predict_from_feature_set(self, feature_set):
        features_set_norm = normalize(feature_set, norm="l2")

        action_pred_proba = self.classifier.predict_proba(features_set_norm)

        max_val_index = np.unravel_index(
            action_pred_proba.argmax(), action_pred_proba.shape
        )

        action_pred = max_val_index[1]

        self.buffer.append(action_pred)
        freq_pred = stats.mode(self.buffer)[0]

        return action_pred, freq_pred[0]

    def predict(self, n_features_sets):
        if len(n_features_sets.shape) == 3:
            y_pred = np.zeros(n_features_sets.shape[2])

            n = n_features_sets.shape[2]

            for i in range(n):
                features_set = n_features_sets[:, :, i]

                action_pred, freq_pred = self._predict_from_feature_set(features_set)

                if self.use_action:
                    y_pred[i] = action_pred
                else:
                    y_pred[i] = freq_pred

        elif len(n_features_sets.shape) == 2:
            y_pred = np.zeros(1)

            features_set = n_features_sets

            action_pred, freq_pred = self._predict_from_feature_set(features_set)

            if self.use_action:
                y_pred[0] = action_pred
            else:
                y_pred[0] = freq_pred
        else:
            raise ValueError("Expected 2D array")

        return y_pred
