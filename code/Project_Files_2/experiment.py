import time
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
# matplotlib.use("Qt5Agg")
np.set_printoptions(precision=3, linewidth=200)

# Data and Model Version
SUFFIX = 'final' # all-persons | all-actions | all-backgrounds


def generate_data(sequence):
    Xtrain = np.zeros((1, config.NUM_HU))
    ytrain = np.zeros(1)

    counter = 0

    for person_num in sequence[:]:
        for action in list(actions.keys())[:]:
            for background in backgrounds[:1]:

                action_video = ActionVideo(person_num, action, background)
                print(action_video)

                action_video.analyze_frames()

                Xtrain = np.vstack((Xtrain, action_video.frame_features))
                ytrain = np.hstack((ytrain, action_video.frame_labels.reshape(-1)))

                counter += 1

    print(f"Total videos analyzed: {counter}")

    return Xtrain[1:], ytrain[1:]


def generate_data_and_train_classifier(use_grid_search=False):
    print("Generate data ...")
    X_train, y_train = generate_data(config.training_sequence + config.validation_sequence)
    X_test, y_test = generate_data(config.test_sequence)

    print("Normalizing data ...")
    x_train_norm = normalize(X_train, norm="l2")

    print("Training classifier ...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')

    if use_grid_search:
        print("Applying Grid Search ...")
        clf = RandomForestClassifier(random_state=42)
        parameters = {
            "n_estimators": [5, 50, 100, 150, 200],
            "max_depth": [None, 5, 10, 50, 100, 500, 1000],
        }
        clf = GridSearchCV(clf, parameters, cv=10, refit=True)

    clf.fit(x_train_norm, y_train)

    print("Saving data and classifier ...")
    np.save(f"{SAVED_DATA_DIR}/X_train_{SUFFIX}", X_train)
    np.save(f"{SAVED_DATA_DIR}/y_train_{SUFFIX}", y_train)

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
    x_test_norm, y_test = load_data("test")
    print(y_train.shape)
    clf = pickle.load(open(f"saved_objects/actions_rfc_model_{SUFFIX}.pkl", "rb"))

    print("Predicting ...")
    y_train_predicted = clf.predict(x_train_norm)
    training_accuracy = accuracy_score(y_train, y_train_predicted)
    print(f"\nTraining set accuracy: {training_accuracy}")

    y_random = np.random.choice(list(range(8)), size=y_test.shape, replace=True)
    baseline_accuracy = accuracy_score(y_test, y_random)
    print(f"\nTest set accuracy baseline (randomized pick): {baseline_accuracy}")

    y_test_predicted = clf.predict(x_test_norm)
    test_accuracy = accuracy_score(y_test, y_test_predicted)
    print(f"\nTest set accuracy: {test_accuracy}")

    if show_confusion_matrix:
        class_names = np.array(
            ["no action", "boxing", "clapping", "waving", "jogging", "running", "walking",]
        )

        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

        plot_confusion_matrix(
            clf,
            x_train_norm,
            y_train,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize="true",
            ax=ax1,
        )
        ax1.set_title("Confusion Matrix - Training Data")

        plot_confusion_matrix(
            clf,
            x_test_norm,
            y_test,
            display_labels=class_names,
            cmap=plt.cm.Greens,
            normalize="true",
            ax=ax2,
        )
        ax2.set_title("Confusion Matrix - Test Data")

        f.tight_layout()
        filename = f"{OUTPUT_DIR}/confusion_matrix_naive_{SUFFIX}.png"
        f.savefig(filename)
        print(f"\n******* Saved image to {filename}")
        plt.show() #TODO remove


def plot_roc_curve(
    x_train_norm,
    y_train_binarized,
    x_test_norm,
    y_test_binarized,
    n_classes,
    clf,
    title,
    ax
):
    classifier = OneVsRestClassifier(clf)
    # print(classifier)
    y_test_score = classifier.fit(x_train_norm, y_train_binarized).predict_proba(
        x_test_norm
    )

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_test_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lw = 2
    for i in range(n_classes):
        ax.plot(
            fpr[i], tpr[i], lw=lw, label=f"{config.labels[i]} (area = {roc_auc[i]:.2f})"
        )
    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC {title}")
    ax.legend(loc="lower right")

    # filename = f"{OUTPUT_DIR}/roc_{title}_{SUFFIX}.png"
    # plt.savefig(filename)
    # print(f"\n******* Saved image to {filename}")
    # plt.show()  # TODO remove


def generate_roc_curves():
    x_train_norm, y_train = load_data("train")
    x_test_norm, y_test = load_data("test")

    y_train_binarized = label_binarize(y_train, classes=list(range(7)))
    y_test_binarized = label_binarize(y_test, classes=list(range(7)))
    n_classes = y_test_binarized.shape[1]

    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
    # f.tight_layout

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')
    title = "random_forest"
    plot_roc_curve(x_train_norm, y_train_binarized, x_test_norm, y_test_binarized, n_classes, clf, title, ax1)

    clf = GaussianNB()
    title = "naive_bayes"
    plot_roc_curve(x_train_norm,y_train_binarized,x_test_norm,y_test_binarized,n_classes,clf,title,ax2)

    clf = KNeighborsClassifier(n_neighbors=3)
    title = "knn"
    plot_roc_curve(x_train_norm,y_train_binarized,x_test_norm,y_test_binarized,n_classes,clf,title,ax3)

    f.subplots_adjust(wspace=0.3)
    filename = f"{OUTPUT_DIR}/roc_classifier_comparison_{SUFFIX}.png"
    f.savefig(filename)
    print(f"\n******* Saved image to {filename}")

def _generate_data_lookback_tau(sequence, clf):
    X = np.zeros((1, config.NUM_HU))
    y = np.zeros(1)

    X_set = np.zeros((1, config.NUM_WINDOWS, config.NUM_HU))

    counter = 0

    for num in sequence[:]:
        for action in list(actions.keys())[:]:
            for background in backgrounds[:1]:
                key_name = f"person{num:02d}_{action}_{background}"
                filename = f"{key_name}_uncomp.avi"

                action_video = ActionVideoUnknownTau(clf, filename, action)
                print(action_video)

                X = np.vstack((X, action_video.frame_features))
                y = np.hstack((y, action_video.frame_labels.reshape(-1)))

                X_set = np.append(X_set, action_video.n_features_sequence, axis=0)

                counter += 1

    print(f"Total videos analyzed: {counter}")

    return X[1:], y[1:], X_set[1:, :, :]


def compare_backward_looking_tau_accuracy(generate_data=False, show_plot=False):
    class_names = np.array(np.array([v for k,v in config.labels.items()]))
    labels = np.array(list(range(7)))

    clf = pickle.load(open(f"saved_objects/actions_rfc_model_{SUFFIX}.pkl", "rb"))

    if generate_data:
        sequence = config.test_sequence
        frame_features, frame_labels, n_features_sequence = _generate_data_lookback_tau(sequence, clf)
        print("Saving lookback tau data")
        np.save(f"{SAVED_DATA_DIR}/frame_features_max_tau_{SUFFIX}", frame_features)
        np.save(f"{SAVED_DATA_DIR}/frame_labels_max_tau_{SUFFIX}", frame_labels)
        np.save(f"{SAVED_DATA_DIR}/n_features_sequence_lookback_tau_{SUFFIX}", n_features_sequence)
    else:
        print("Loading lookback tau data")
        frame_features  = np.load(f"{SAVED_DATA_DIR}/frame_features_max_tau_{SUFFIX}.npy")
        frame_labels  = np.load(f"{SAVED_DATA_DIR}/frame_labels_max_tau_{SUFFIX}.npy")
        n_features_sequence  = np.load(f"{SAVED_DATA_DIR}/n_features_sequence_lookback_tau_{SUFFIX}.npy")

    modified_random_forest_clf_action = ModifiedRandomForest(clf, use_action=True)
    modified_random_forest_clf_freq = ModifiedRandomForest(clf, buffer_len=10)

    if show_plot:
        f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(24, 6))

        title = "Confusion Matrix - Fixed \u03C4 (Max)"
        print("Starting predict fixed tau")
        plot_confusion_matrix(
            clf,
            frame_features,
            frame_labels,
            labels=labels,
            display_labels=class_names,
            cmap=plt.cm.Purples,
            normalize="true",
            ax=ax1
        )
        ax1.set_title(title)

        title = "Confusion Matrix - Lookback \u03C4"
        print("Starting predict lookback tau")
        plot_confusion_matrix(
            modified_random_forest_clf_action,
            n_features_sequence,
            frame_labels,
            labels=labels,
            display_labels=class_names,
            cmap=plt.cm.Oranges,
            normalize="true",
            ax=ax2
        )
        ax2.set_title(title)

        title = "Confusion Matrix - Lookback \u03C4 with frequency counter"
        print("Starting predict lookback tau with frequency buffer")
        plot_confusion_matrix(
            modified_random_forest_clf_freq,
            n_features_sequence,
            frame_labels,
            labels=labels,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize="true",
            ax=ax3
        )
        ax3.set_title(title)

        f.tight_layout()
        filename = f"{OUTPUT_DIR}/confusion_matrix_lookback_tau_{SUFFIX}.png"
        f.savefig(filename)
        print(f"\n******* Saved image to {filename}")
        plt.show() # TODO remove

    start = time.time()
    fixed_tau_labels_predicted = clf.predict(frame_features)
    fixed_tau_score = accuracy_score(frame_labels, fixed_tau_labels_predicted)
    print(f"\nFixed tau accuracy: {fixed_tau_score}\truntime: {time.time() - start}")

    start = time.time()
    lookback_tau_labels_predicted = modified_random_forest_clf_action.predict(n_features_sequence)
    lookback_tau_score = accuracy_score(frame_labels, lookback_tau_labels_predicted)
    print(f"\nTest set accuracy baseline (randomized pick): {lookback_tau_score}\truntime: {time.time() - start}")

    start = time.time()
    lookback_tau_freq_labels_predicted = modified_random_forest_clf_freq.predict(n_features_sequence)
    lookback_tau_freq_score = accuracy_score(frame_labels, lookback_tau_freq_labels_predicted)
    print(f"\nTest set accuracy baseline (randomized pick): {lookback_tau_freq_score}\truntime: {time.time() - start}")


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

    filename = f"{OUTPUT_DIR}/hu_moments_by_action_{SUFFIX}.png"
    fig.savefig(filename)
    print(f"\n******* Saved image to {filename}")
    plt.show() # TODO Remove.


def label_final_spliced_action_video():
    filename = "spliced_action_video.mp4"

    clf = pickle.load(open(f"saved_objects/actions_rfc_model_{SUFFIX}.pkl", "rb"))
    modified_clf = ModifiedRandomForest(clf, buffer_len=15)

    frames_to_save = [50, 150, 250, 350, 450, 650, 750, 850, 950]
    live_action_video = VideoActionLabeler(modified_clf, filename, 25)
    live_action_video.create_annotated_video(frame_ids=frames_to_save)


def process_cmdline_args():
    """Processes command line arguments"""
    desc = """Runs scripts used to generate output in the report.

        NOTE: Large data files required to run some experiments are not included in the
              package. They can be downloaded from https://alksjdflsj.com

        exp 0:  Desc: Generates data, trains classifier, and saves both to disk.
                Runtime: ~ 1 hr
                Requires:
                   ./input_files directory with all raw video dataset.

        exp 1:  Desc: Compares classifier accuracy and generates confusion matrix plots
                Runtime: ~ 1 min
                Requires:
                   ./saved_objects directory with trained model and saved training/testing data.

        exp 2:  Desc: Generates plots for hu-moments by frame to for each action.
                      Defaults to person 10 over d1 background in dataset.
                Runtime: ~ 30 secs
                Requires:
                   ./input_files person 10 action videos

        exp 3:  Desc: Generates ROC curves for various classifiers.
                Runtime: ~ 4 min
                Requires:
                   ./saved_objects directory with trained model and saved training/testing data.

        exp 4:  Desc: Compare fixed tau prediction with backward looking tau.
                Runtime: ~
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
            "Uncomment this next line and run again."
            "\n# generate_data_and_train_classifier()"
        )
        # generate_data_and_train_classifier()

    elif args.exp == "1":
        compare_classifier_accuracy(show_confusion_matrix=True)

    elif args.exp == "2":
        generate_plots_for_different_actions()

    elif args.exp == "3":
        generate_roc_curves()

    elif args.exp == "4":
        compare_backward_looking_tau_accuracy(generate_data=False)

    elif args.exp == "5":
        label_final_spliced_action_video()
