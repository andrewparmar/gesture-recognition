import pprint
import warnings
import pickle

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.utils.multiclass import unique_labels

import config
import core
from core import ActionVideo, TAU, NUM_HU, InputActionVideo, LiveActonVideo, VID_DIR, OUTPUT_DIR, ModifiedRandomForest

warnings.filterwarnings("ignore")

# Console settings
matplotlib.use("Qt5Agg")

np.set_printoptions(precision=3, linewidth=200)

from config import SAVED_DATA_DIR
SUFFIX = '0415'
# SUFFIX = '0414_test'


def label_final_spliced_action_video():
    clf = pickle.load(open('saved_objects/actions_rfc_model.pkl', 'rb'))

    filename = "spliced_action_video.mp4"
    live_action_video = LiveActonVideo(clf, filename, 25)
    live_action_video.create_annotated_video()


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

    clf = pickle.load(open(f'saved_objects/actions_rfc_model_{SUFFIX}.pkl', 'rb'))

    print("Predicting ...")
    y_train_predicted = clf.predict(x_train_norm)
    training_accuracy = accuracy_score(y_train, y_train_predicted)
    print(f"\nTraining set accuracy: {training_accuracy}")

    y_random = np.random.choice(list(range(8)), size=y_validation.shape, replace=True,
                                p=None)
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
            ["blank", "boxing", "clapping", "waving", "jogging", "running", "walking", ]
        )

        title = "Confusion Matrix - Training Data"

        disp = plot_confusion_matrix(
            clf,
            x_train_norm,
            y_train,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize="true",
        )
        disp.figure_.savefig(f"{OUTPUT_DIR}/confusion_matrix_training.png")

        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
        plt.show()

        ##################################################################################

        title = "Confusion Matrix - Validation Data"

        disp = plot_confusion_matrix(
            clf,
            x_validation_norm,
            y_validation,
            display_labels=class_names,
            cmap=plt.cm.Reds,
            normalize="true",
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

        plt.show()

        disp.figure_.savefig(f"{OUTPUT_DIR}/confusion_matrix_validation.png")


def generate_data_and_train_classifier():
    # Generate frame features fom video files.
    print("Generate data ...")
    X_train, y_train = core.generate_data(config.training_sequence)
    X_validation, y_validation = core.generate_data(config.validation_sequence)
    X_test, y_test = core.generate_data(config.test_sequence)

    # Save the data
    np.save(f"{SAVED_DATA_DIR}/X_train_{SUFFIX}", X_train)
    np.save(f"{SAVED_DATA_DIR}/y_train_{SUFFIX}", y_train)

    np.save(f"{SAVED_DATA_DIR}/X_validation_{SUFFIX}", X_validation)
    np.save(f"{SAVED_DATA_DIR}/y_validation_{SUFFIX}", y_validation)

    np.save(f"{SAVED_DATA_DIR}/X_test_{SUFFIX}", X_test)
    np.save(f"{SAVED_DATA_DIR}/y_test_{SUFFIX}", y_test)

    # Normalize the data
    print("Normalizing data ...")
    x_train_norm = normalize(X_train, norm="l2")

    print("Training classifier ...")
    # clf = RandomForestClassifier(n_estimators=100, random_state=42)

    ######################################################################################
    clf = RandomForestClassifier(random_state=42)
    parameters = {'n_estimators': [5, 50, 100, 150, 200],
                  'max_depth': [None, 10, 50, 100, 500, 1000]}
    clf = GridSearchCV(clf, parameters, cv=10, refit=True)
    ######################################################################################

    clf.fit(x_train_norm, y_train)
    pickle.dump(clf, open(f'saved_objects/actions_rfc_model_{SUFFIX}.pkl', 'wb'))

def compare_backward_looking_tau_accuracy(filename):
    class_names = np.array(
        ["blank", "boxing", "clapping", "waving", "jogging", "running", "walking", ]
    )
    labels = np.array(list(range(7)))

    clf = pickle.load(open('saved_objects/actions_rfc_model.pkl', 'rb'))

    input_action_video = InputActionVideo(clf, filename, 'running') # TODO Why are you proviiding the filename here??

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

if __name__ == "__main__":
    #todo Add argparse
    # get_data = False
    # train = False
    # show_graph = False

    generate_data_and_train_classifier()

    compare_classifier_accuracy(show_confusion_matrix=True)

    # label_final_spliced_action_video()

    # compare_backward_looking_tau_accuracy(filename = f"person19_running_d1_uncomp.avi")


