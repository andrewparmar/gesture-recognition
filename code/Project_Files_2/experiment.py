import os
import pickle
import warnings
from argparse import ArgumentParser, RawTextHelpFormatter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, plot_confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize, normalize

import config
from config import OUTPUT_DIR, SAVED_DATA_DIR, actions, backgrounds
from core import (
    ActionVideo,
    ActionVideoUnknownTau,
    ModifiedRandomForest,
    VideoActionLabeler,
)

# Console settings
warnings.filterwarnings("ignore")
matplotlib.use("Qt5Agg")
np.set_printoptions(precision=3, linewidth=200)

# Data and Model Version
# SUFFIX = '0415_0' # all-persons | all-actions | d1
SUFFIX = "0415_1"  # all-persons | all-actions | all-backgrounds
# SUFFIX = "0415_2"  # 1-persons | 1actions | d1
# SUFFIX = '0415_3' # all-persons | all-actions | all-backgrounds | Grid Search

# SUFFIX = '0417_0' # all-persons | all-actions | d1 |

# SUFFIX = 'Final' # all-persons | all-actions | all-backgrounds


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

    if use_grid_search:
        print("Applying Grid Search ...")
        clf = RandomForestClassifier(random_state=42)
        parameters = {
            "n_estimators": [5, 50, 100, 150, 200],
            "max_depth": [None, 10, 50, 100, 500, 1000],
        }
        clf = GridSearchCV(clf, parameters, cv=10, refit=True)

    clf.fit(x_train_norm, y_train)

    print("Saving data and classifier ...")
    np.save(f"{SAVED_DATA_DIR}/X_train_{SUFFIX}", X_train)
    np.save(f"{SAVED_DATA_DIR}/y_train_{SUFFIX}", y_train)

    np.save(f"{SAVED_DATA_DIR}/X_validation_{SUFFIX}", X_validation)
    np.save(f"{SAVED_DATA_DIR}/y_validation_{SUFFIX}", y_validation)

    np.save(f"{SAVED_DATA_DIR}/X_test_{SUFFIX}", X_test)
    np.save(f"{SAVED_DATA_DIR}/y_test_{SUFFIX}", y_test)
    pickle.dump(clf, open(f"saved_objects/actions_rfc_model_{SUFFIX}.pkl", "wb"))


def load_data(data_type):
    print(f"Loading and normalizing {data_type} data ...")
    X_ = np.load(f"{SAVED_DATA_DIR}/X_{data_type}_{SUFFIX}.npy")
    y_ = np.load(f"{SAVED_DATA_DIR}/y_{data_type}_{SUFFIX}.npy")

    x_norm = normalize(X_, norm="l2")

    return x_norm, y_


def compare_classifier_accuracy(show_confusion_matrix=False):
    # Load the data
    x_train_norm, y_train = load_data("train")
    x_validation_norm, y_validation = load_data("validation")
    x_test_norm, y_test = load_data("test")

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
        disp.figure_.savefig(f"{OUTPUT_DIR}/confusion_matrix_training-{SUFFIX}.png")

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
        ax2.set_title(title)

        print(title)
        print(disp.confusion_matrix)

        disp.figure_.savefig(f"{OUTPUT_DIR}/confusion_matrix_validation-{SUFFIX}.png")

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
        ax3.set_title(title)

        print(title)
        print(disp.confusion_matrix)

        disp.figure_.savefig(f"{OUTPUT_DIR}/confusion_matrix_test-{SUFFIX}.png")

        plt.show()


def plot_roc_curve(
    x_train_norm,
    y_train_binarized,
    x_test_norm,
    y_test_binarized,
    n_classes,
    clf,
    title,
):
    classifier = OneVsRestClassifier(clf)
    y_test_score = classifier.fit(x_train_norm, y_train_binarized).predict_proba(
        x_test_norm
    )

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_test_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    for i in range(n_classes):
        plt.plot(
            fpr[i], tpr[i], lw=lw, label=f"{config.labels[i]} (area = {roc_auc[i]:.2f})"
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC {title}")
    plt.legend(loc="lower right")
    plt.savefig(f"{OUTPUT_DIR}/roc_{title}_{SUFFIX}.svg", format="svg")


def generate_roc_curves():
    x_train_norm, y_train = load_data("train")
    x_test_norm, y_test = load_data("test")

    y_train_binarized = label_binarize(y_train, classes=list(range(7)))
    y_test_binarized = label_binarize(y_test, classes=list(range(7)))
    n_classes = y_test_binarized.shape[1]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    title = "random_forest"
    plot_roc_curve(
        x_train_norm,
        y_train_binarized,
        x_test_norm,
        y_test_binarized,
        n_classes,
        clf,
        title,
    )

    clf = GaussianNB()
    title = "naive_bayes"
    plot_roc_curve(
        x_train_norm,
        y_train_binarized,
        x_test_norm,
        y_test_binarized,
        n_classes,
        clf,
        title,
    )

    clf = KNeighborsClassifier(n_neighbors=3)
    title = "knn"
    plot_roc_curve(
        x_train_norm,
        y_train_binarized,
        x_test_norm,
        y_test_binarized,
        n_classes,
        clf,
        title,
    )


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
    disp.figure_.savefig(f"{OUTPUT_DIR}/confusion_matrix_training_{SUFFIX}.png")

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
    disp.figure_.savefig(f"{OUTPUT_DIR}/confusion_matrix_training_{SUFFIX}.png")

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
    disp.figure_.savefig(f"{OUTPUT_DIR}/confusion_matrix_training_{SUFFIX}.png")

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


def label_final_spliced_action_video():
    filename = "spliced_action_video.mp4"

    clf = pickle.load(open(f"saved_objects/actions_rfc_model_{SUFFIX}.pkl", "rb"))
    modified_clf = ModifiedRandomForest(clf, buffer_len=10)

    frames_to_save = [50, 150, 250, 350, 450, 650, 750, 850, 950]
    live_action_video = VideoActionLabeler(modified_clf, filename, 25)
    live_action_video.create_annotated_video(frame_ids=frames_to_save)


def process_cmdline_args():
    """Processes command line arguments"""
    desc = """Runs scripts used to generate output in the report.

        NOTE: Large data files required to run some experiments are not included in the
              package. They can be downloaded from https://alksjdflsj.com

        exp 0:  Runtime: ~ 1 hr
                Desc: Generates data, trains classifier, and saves both to disk.
                Requires:
                   ./input_files directory with all raw video dataset.

        exp 1:  Runtime: ~ 1 min
                Desc: Compares prediction accuracies and generates confusion matrix plots
                Requires:
                   ./saved_objects directory with trained model and saved training/testing data.

        exp 2:  Runtime: ~ 30 secs
                Desc: Generates plots for hu-moments by frame to for each action.
                      Defaults to person 10 over d1 background in dataset.
                Requires:
                   ./input_files person 10 action videos

        exp 3:  Runtime: ~ 4 min
                Desc: Generates ROC curves for various classifiers.
                Requires:
                   ./saved_objects directory with trained model and saved training/testing data.

        exp 4:  Runtime: ~
                Desc: Compare fixed tau prediction with backward looking tau.
                Requires:
                   ./saved_objects/....
                   ./saved_objects/....
                   ./saved_objects/....
                   ./saved_objects/....

        exp 5:  Runtime: ~ 5 mins
                Desc: Analyzes video frames and outputs video with overlay action labels.
                Requires:
                   ./saved_objects directory with trained model.
    """
    examples = """
    examples:
        Generate hu moments plots for all actions
            python experiment.py --exp 2
    """
    parser = ArgumentParser(
        description=desc, epilog=examples, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("--exp", help="experiment number", default="exp1", type=str)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    try:
        os.mkdir(OUTPUT_DIR)
    except OSError:
        pass

    args = process_cmdline_args()

    if args.exp == "0":
        print(
            "Are you sure about this? This takes a while. "
            "Uncomment the next line and run again."
        )
        # generate_data_and_train_classifier()

    elif args.exp == "1":
        compare_classifier_accuracy(show_confusion_matrix=True)

    elif args.exp == "2":
        generate_plots_for_different_actions()

    elif args.exp == "3":
        generate_roc_curves()

    elif args.exp == "4":
        compare_backward_looking_tau_accuracy(filename=f"person19_running_d1_uncomp.avi")

    elif args.exp == "5":
        label_final_spliced_action_video()
