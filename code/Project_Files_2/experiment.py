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
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils.multiclass import unique_labels

import config
import core
from core import ActionVideo, TAU, NUM_HU, InputActionVideo, LiveActonVideo

warnings.filterwarnings("ignore")


# Console settings
matplotlib.use("Qt5Agg")

np.set_printoptions(precision=3, linewidth=200)

SAVED_DATA_DIR = "saved_objects"


if __name__ == "__main__":
    get_data = False
    show_graph = False
    train = False

    if get_data:
        # X_train, y_train = core.generate_data(config.training_sequence)
        X_validation, y_validation = core.generate_data(config.validation_sequence)
        # X_test, y_test = core.generate_data(config.test_sequence)

        # Save the data
        # np.save(f"{SAVED_DATA_DIR}/X_train_{TAU}", X_train)
        # np.save(f"{SAVED_DATA_DIR}/y_train_{TAU}", y_train)

        np.save(f"{SAVED_DATA_DIR}/X_validation_{TAU}", X_validation)
        np.save(f"{SAVED_DATA_DIR}/y_validation_{TAU}", y_validation)

        # np.save(f"{SAVED_DATA_DIR}/X_test_{TAU}", X_test)
        # np.save(f"{SAVED_DATA_DIR}/y_test_{TAU}", y_test)

    # Load the data
    print("Loading data ...")
    X_train = np.load(f"{SAVED_DATA_DIR}/X_train_{TAU}.npy")
    y_train = np.load(f"{SAVED_DATA_DIR}/y_train_{TAU}.npy")

    X_validation = np.load(f"{SAVED_DATA_DIR}/X_validation_{TAU}.npy")
    y_validation = np.load(f"{SAVED_DATA_DIR}/y_validation_{TAU}.npy")

    # X_test = np.load(f"{SAVED_DATA_DIR}/X_test_{TAU}.npy")
    # y_test = np.load(f"{SAVED_DATA_DIR}/y_test_{TAU}.npy")

    # Normalize the data
    print("Normalizing data ...")
    x_train_norm = normalize(X_train, norm="l2")
    x_validation_norm = normalize(X_validation, norm="l2")
    # x_test_norm = normalize(X_test, norm="l2")

    if train:
        print("Training classifier ...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(x_train_norm, y_train)
        pickle.dump(clf, open('saved_objects/actions_rfc_model.pkl', 'wb'))
    else:
        clf = pickle.load(open('saved_objects/actions_rfc_model.pkl', 'rb'))

    ######################################################################################
    # parameters = {'n_estimators': [50, 100, 150],
    #               'max_depth': [None, 10, 50, 100, 500, 1000]}
    # clf = GridSearchCV(clf, parameters, cv=10, refit=True)
    #
    # clf.fit(X_train, y_train)
    ######################################################################################

    print("Predicting ...")
    y_train_predicted = clf.predict(x_train_norm)
    training_accuracy = accuracy_score(y_train, y_train_predicted)
    print(f"\nTraining set accuracy: {training_accuracy}")

    y_random = np.random.choice(list(range(8)), size=y_validation.shape, replace=True, p=None)
    baseline_accuracy = accuracy_score(y_validation, y_random)
    print(f"\nBaseline accuracy: {baseline_accuracy}")

    y_validation_predicted = clf.predict(x_validation_norm)
    validation_accuracy = accuracy_score(y_validation, y_validation_predicted)
    print(f"\nValidation set accuracy: {validation_accuracy}")


    y_test_predictions = []

    filename = f"person19_running_d1_uncomp.avi"
    # input_action_video = InputActionVideo(clf, filename, 'jogging')
    # input_action_video.play(clf)

    # # try:
    # buffer = np.zeros(5, dtype=np.uint8)
    # for i, feature_set in enumerate(input_action_video.frame_feature_set_generator()):
    #     try:
    #         features_set_norm = normalize(feature_set, norm="l2")
    #     except:
    #         import pdb; pdb.set_trace()
    #
    #
    #     # try:
    #     #     assert np.all(features_set_norm == x_test_norm[i])
    #     # except:
    #     #     print("inside here ***************** ")
    #     #     # import pdb; pdb.set_trace()
    #     #     pass
    #
    #     # action_pred = clf.predict(features_set_norm[20].reshape(1, -1)).astype(np.uint8)
    #     action_pred_proba = clf.predict_proba(features_set_norm)
    #     action_pred = np.unravel_index(action_pred_proba.argmax(), action_pred_proba.shape)[1]
    #
    #     # import pdb; pdb.set_trace()
    #
    #
    #     # counts = np.bincount(action_pred)
    #     # prediction = np.argmax(counts)
    #     # predict_other = clf.predict(x_validation_norm[i].reshape(1, -1))
    #
    #     # print(action_pred, predict_other)
    #     # import pdb; pdb.set_trace()
    #     buffer = np.hstack((buffer, action_pred))
    #     print(buffer[-6:])
    #     freq_pred = np.argmax(np.bincount(buffer[-6:]))
    #     print(action_pred, freq_pred)
    #
    #
    #     # if predict_other == 1:
    #     #     import pdb; pdb.set_trace()
    #
    #     y_test_predictions.append(action_pred)
    #     # pass
    # # except Exception as e:
    # #     import pdb; pdb.set_trace()

    live_action_video = LiveActonVideo(clf, filename, 25)
    live_action_video.create_annotated_video()

    if show_graph:
        cm = confusion_matrix(y_validation, y_validation_predicted)

        class_names = np.array(
            ["blank", "boxing", "clapping", "waving", "jogging", "running", "walking",]
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
        disp.figure_.savefig("confusion_matrix_training.png")

        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

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

        disp.figure_.savefig("confusion_matrix_testing.png")