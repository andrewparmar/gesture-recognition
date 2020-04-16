import os
import pickle
import warnings
from argparse import ArgumentParser, RawTextHelpFormatter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize

import config
from config import OUTPUT_DIR, SAVED_DATA_DIR, actions, backgrounds
from core import (
    ActionVideo,
    ActionVideoUnknownTau,
    ModifiedRandomForest,
    VideoActionLabeler,
)
from utils import plot_features

# Console settings
warnings.filterwarnings("ignore")
matplotlib.use("Qt5Agg")
np.set_printoptions(precision=3, linewidth=200)


# SUFFIX = '0415_0' # all-persons | all-actions | d1
# SUFFIX = '0415_1' # all-persons | all-actions | all-backgrounds
SUFFIX = "0415_2"  # 1-persons | 1actions | d1
# SUFFIX = '0415_3' # all-persons | all-actions | all-backgrounds | Grid Search
# SUFFIX = 'Final'


def generate_data(sequence):
    Xtrain = np.zeros((1, config.NUM_HU))
    ytrain = np.zeros(1)

    for person_num in sequence[:1]:
        for action in list(actions.keys())[:1]:
            for background in backgrounds[:1]:

                action_video = ActionVideo(person_num, action, background)
                print(action_video)

                action_video.analyze_frames()

                Xtrain = np.vstack((Xtrain, action_video.frame_features))
                ytrain = np.hstack((ytrain, action_video.frame_labels.reshape(-1)))

    # print(f"Average time {sum(times)/len(times)}")

    return Xtrain[1:], ytrain[1:]


def generate_data_and_train_classifier(use_grid_search=False):
    print("Generate data ...")
    X_train, y_train = generate_data(config.training_sequence)
    X_validation, y_validation = generate_data(config.validation_sequence)
    X_test, y_test = generate_data(config.test_sequence)

    print("Normalizing data ...")
    x_train_norm = normalize(X_train, norm="l2")

    print("Training classifier ...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    ######################################################################################
    if use_grid_search:
        print("Applying Grid Search ...")
        clf = RandomForestClassifier(random_state=42)
        parameters = {
            "n_estimators": [5, 50, 100, 150, 200],
            "max_depth": [None, 10, 50, 100, 500, 1000],
        }
        clf = GridSearchCV(clf, parameters, cv=10, refit=True)
    ######################################################################################

    clf.fit(x_train_norm, y_train)

    print("Saving data and classifier ...")
    np.save(f"{SAVED_DATA_DIR}/X_train_{SUFFIX}", X_train)
    np.save(f"{SAVED_DATA_DIR}/y_train_{SUFFIX}", y_train)

    np.save(f"{SAVED_DATA_DIR}/X_validation_{SUFFIX}", X_validation)
    np.save(f"{SAVED_DATA_DIR}/y_validation_{SUFFIX}", y_validation)

    np.save(f"{SAVED_DATA_DIR}/X_test_{SUFFIX}", X_test)
    np.save(f"{SAVED_DATA_DIR}/y_test_{SUFFIX}", y_test)
    pickle.dump(clf, open(f"saved_objects/actions_rfc_model_{SUFFIX}.pkl", "wb"))


def compare_classifier_accuracy(show_confusion_matrix=False):
    # Load the data
    print("Loading data ...")
    X_train = np.load(f"{SAVED_DATA_DIR}/X_train_{SUFFIX}.npy")
    y_train = np.load(f"{SAVED_DATA_DIR}/y_train_{SUFFIX}.npy")

    X_validation = np.load(f"{SAVED_DATA_DIR}/X_validation_{SUFFIX}.npy")
    y_validation = np.load(f"{SAVED_DATA_DIR}/y_validation_{SUFFIX}.npy")

    X_test = np.load(f"{SAVED_DATA_DIR}/X_test_{SUFFIX}.npy")
    y_test = np.load(f"{SAVED_DATA_DIR}/y_test_{SUFFIX}.npy")

    # Normalize the data
    print("Normalizing data ...")
    x_train_norm = normalize(X_train, norm="l2")
    x_validation_norm = normalize(X_validation, norm="l2")
    x_test_norm = normalize(X_test, norm="l2")

    clf = pickle.load(open(f"saved_objects/actions_rfc_model_{SUFFIX}.pkl", "rb"))

    print("Predicting ...")
    y_train_predicted = clf.predict(x_train_norm)
    training_accuracy = accuracy_score(y_train, y_train_predicted)
    print(f"\nTraining set accuracy: {training_accuracy}")

    y_random = np.random.choice(list(range(8)), size=y_validation.shape, replace=True)
    baseline_accuracy = accuracy_score(y_validation, y_random)
    print(f"\nBaseline validation set randomized pick accuracy: {baseline_accuracy}")

    y_validation_predicted = clf.predict(x_validation_norm)
    validation_accuracy = accuracy_score(y_validation, y_validation_predicted)
    print(f"\nValidation set accuracy: {validation_accuracy}")

    y_test_predicted = clf.predict(x_test_norm)
    test_accuracy = accuracy_score(y_test, y_test_predicted)
    print(f"\nTest set accuracy: {test_accuracy}")

    if show_confusion_matrix:
        class_names = np.array(
            ["blank", "boxing", "clapping", "waving", "jogging", "running", "walking",]
        )

        f, (ax1, ax2, ax3) = plt.subplots(
            nrows=1, ncols=3, sharey=True, figsize=(24, 4)
        )
        f.tight_layout()

        title = "Confusion Matrix - Training Data"

        disp = plot_confusion_matrix(
            clf,
            x_train_norm,
            y_train,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize="true",
            ax=ax1,
        )
        disp.figure_.savefig(f"{OUTPUT_DIR}/confusion_matrix_training.png")

        # disp.ax_.set_title(title)
        ax1.set_title(title)

        print(title)
        print(disp.confusion_matrix)

        ##################################################################################

        title = "Confusion Matrix - Validation Data"

        disp = plot_confusion_matrix(
            clf,
            x_validation_norm,
            y_validation,
            display_labels=class_names,
            cmap=plt.cm.Oranges,
            normalize="true",
            ax=ax2,
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

        disp.figure_.savefig(f"{OUTPUT_DIR}/confusion_matrix_validation.png")

        ##################################################################################

        title = "Confusion Matrix - Test Data"

        disp = plot_confusion_matrix(
            clf,
            x_test_norm,
            y_test,
            display_labels=class_names,
            cmap=plt.cm.Greens,
            normalize="true",
            ax=ax3,
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

        disp.figure_.savefig(f"{OUTPUT_DIR}/confusion_matrix_test.png")

        # import pdb; pdb.set_trace()

        # plt.tight_layout()
        plt.show()


def compare_backward_looking_tau_accuracy(filename):
    class_names = np.array(
        ["blank", "boxing", "clapping", "waving", "jogging", "running", "walking",]
    )
    labels = np.array(list(range(7)))

    clf = pickle.load(open("saved_objects/actions_rfc_model.pkl", "rb"))

    input_action_video = ActionVideoUnknownTau(
        clf, filename, "running"
    )  # TODO Why are you providing the filename here??

    title = "Confusion Matrix - Fixed Tau"
    print("Starting predict fixed tau")
    disp = plot_confusion_matrix(
        clf,
        input_action_video.frame_features,
        input_action_video.frame_labels,
        labels=labels,
        display_labels=class_names,
        # cmap=plt.cm.Blues,
        normalize="true",
    )
    disp.figure_.savefig(f"{OUTPUT_DIR}/confusion_matrix_training.png")

    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    # plt.show()

    # input_action_video.play(clf)

    modified_random_forest_clf_freq = ModifiedRandomForest(clf, buffer_len=5)
    modified_random_forest_clf_action = ModifiedRandomForest(clf, use_action=True)

    # y_pred = modified_random_forest_clf.predict(input_action_video.n_features_sequence)

    title = "Confusion Matrix - Backward tau action probability with frequency"
    print("Starting predict freq")
    disp = plot_confusion_matrix(
        modified_random_forest_clf_freq,
        input_action_video.n_features_sequence,
        input_action_video.frame_labels,
        labels=labels,
        display_labels=class_names,
        # cmap=plt.cm.Blues,
        normalize="true",
    )
    disp.figure_.savefig(f"{OUTPUT_DIR}/confusion_matrix_training.png")

    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    # plt.show()

    title = "Confusion Matrix - Backward tau action probability"
    print("Starting predict")
    disp = plot_confusion_matrix(
        modified_random_forest_clf_action,
        input_action_video.n_features_sequence,
        input_action_video.frame_labels,
        labels=labels,
        display_labels=class_names,
        # cmap=plt.cm.Blues,
        normalize="true",
    )
    disp.figure_.savefig(f"{OUTPUT_DIR}/confusion_matrix_training.png")

    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    plt.show()


def generate_plots_for_different_actions(person_num=10, background="d1"):
    person_num = person_num
    background = background

    rows = 2
    cols = 3

    fig, axs, = plt.subplots(nrows=2, ncols=3, figsize=(24, 12))
    fig.tight_layout()

    axs_list = []
    for row in range(rows):
        for col in range(cols):
            axs_list.append(axs[row, col])

    for action in config.actions:
        action_video = ActionVideo(person_num, action, background)
        print(action_video)
        action_video.analyze_frames()
        action_video.plot_features_by_frame(axs_list.pop())

    plt.subplots_adjust(
        top=0.927, bottom=0.063, left=0.035, right=0.969, hspace=0.322, wspace=0.15
    )

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/hu_moments_by_action_{SUFFIX}.svg", format="svg")
    # plt.show()


def label_final_spliced_action_video():
    filename = "spliced_action_video.mp4"

    clf = pickle.load(open(f"saved_objects/actions_rfc_model_{SUFFIX}.pkl", "rb"))
    modified_clf = ModifiedRandomForest(clf, buffer_len=10)

    live_action_video = VideoActionLabeler(modified_clf, filename, 25)
    live_action_video.create_annotated_video()


def process_cmdline_args():
    """Processes command line arguments"""
    desc = """Runs scripts used to generate output in the report.

        NOTE: Large data files required to run some experiments are not included in the
              package. They can be downloaded from https://alksjdflsj.com

        exp 0:  [Runtime ~ 1 hr]
                Desc: Generates data, trains classifier, and saves both to disk.
                Requires:
                   ./saved_objects/....
                   ./saved_objects/....
                   ./saved_objects/....
                   ./saved_objects/....

        exp 1:  [Runtime 1 min]
                Desc: Compares prediction accuracies and generates confusion matrix plots
                Requires:
                   ./saved_objects/....
                   ./saved_objects/....
                   ./saved_objects/....
                   ./saved_objects/....

        exp 2:  [Runtime ~ 30 secs]
                Desc: Generates plots for hu-moment representation of action over frames.
                      Defaults to person 10 over d1 background in dataset.
                Requires:
                   ./input_videos/person10_boxing_d1_uncomp.avi
                   ./input_videos/person10_handwaving_d1_uncomp.avi
                   ./input_videos/person10_handclapping_d1_uncomp.avi
                   ./input_videos/person10_running_d1_uncomp.avi
                   ./input_videos/person10_jogging_d1_uncomp.avi
                   ./input_videos/person10_walking_d1_uncomp.avi

        exp 3:  [Runtime ~ 1 hr]
                Desc: Generates data, trains classifier, and saves both to disk.
                Requires:
                   ./saved_objects/....
                   ./saved_objects/....
                   ./saved_objects/....
                   ./saved_objects/....

        exp 4:  [Runtime ~ 1 hr]
                Desc: Generates data, trains classifier, and saves both to disk.
                Requires:
                   ./saved_objects/....
                   ./saved_objects/....
                   ./saved_objects/....
                   ./saved_objects/....
    """
    examples = """
    examples:
        Generate confusion matrix plots
            python experiment.py --exp 1
    """
    parser = ArgumentParser(
        description=desc, epilog=examples, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("--exp", help="Experiment number", default="exp1", type=str)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    try:
        os.mkdir(OUTPUT_DIR)
    except OSError:
        pass

    args = process_cmdline_args()

    if args.exp == "1":
        generate_data_and_train_classifier()
        compare_classifier_accuracy(show_confusion_matrix=True)

    elif args.exp == "2":
        generate_plots_for_different_actions()

        # compare_backward_looking_tau_accuracy(filename = f"person19_running_d1_uncomp.avi")

        # label_final_spliced_action_video()
