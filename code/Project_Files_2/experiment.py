import pprint
import warnings

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils.multiclass import unique_labels

import config
import core
from core import ActionVideo, TAU, NUM_HU

# warnings.filterwarnings("ignore")




# Console settings
matplotlib.use("Qt5Agg")

np.set_printoptions(precision=3, linewidth=200)

SAVED_DATA_DIR = "saved_objects"


def run_moment_calculation():
    # filename = "mhi_frame_200_person01_walking_d1.png"
    filename = "output_images/0_mhi.jpg"
    test_image = cv2.imread(filename)

    for type_ in [np.uint8, np.uint16, np.float]:
        print(f'\nType" {type_}')
        test_image = test_image.astype(type_)

        cat = core.HuMoments(test_image[:, :, 0])
        pprint.pprint(cat.values)

        cv2_moments = cv2.moments(test_image[:, :, 0])
        pprint.pprint(cv2.HuMoments(cv2_moments).flatten())


if __name__ == "__main__":
    get_data = False
    show_graph = False

    if get_data:
        X_train, y_train = core.generate_data(
            config.training_sequence + config.validation_sequence
        )
        X_validation, y_validation = core.generate_data(config.validation_sequence)
        X_test, y_test = core.generate_data(config.test_sequence)

        # Save the data
        np.save(f"{SAVED_DATA_DIR}/X_train_{TAU}", X_train)
        np.save(f"{SAVED_DATA_DIR}/X_test_{TAU}", X_test)
        np.save(f"{SAVED_DATA_DIR}/y_train_{TAU}", y_train)
        np.save(f"{SAVED_DATA_DIR}/y_test_{TAU}", y_test)

    # Load the data
    print("Loading data ...")
    X_train = np.load(f"{SAVED_DATA_DIR}/X_train_{TAU}.npy")
    X_test = np.load(f"{SAVED_DATA_DIR}/X_test_{TAU}.npy")
    y_train = np.load(f"{SAVED_DATA_DIR}/y_train_{TAU}.npy")
    y_test = np.load(f"{SAVED_DATA_DIR}/y_test_{TAU}.npy")

    # Normalize the data
    print("Normalizing data ...")
    x_train_norm = normalize(X_train, norm="l2")
    x_test_norm = normalize(X_test, norm="l2")

    print("Training classifier ...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(x_train_norm, y_train)

    ######################################################################################
    # parameters = {'n_estimators': [50, 100, 150],
    #               'max_depth': [None, 10, 50, 100, 500, 1000]}
    # clf = GridSearchCV(clf, parameters, cv=10, refit=True)
    #
    # clf.fit(X_train, y_train)
    ######################################################################################

    print("Predicting ...")
    accuracy = clf.score(x_train_norm, y_train)
    print(f"\nTraining set accuracy: {accuracy}")

    # Todo: Add The baseline predictions use random choice.
    # baseline_preds = test_features[:, feature_list.index('average')]
    y_random = np.random.choice(list(range(8)), size=y_test.shape, replace=True, p=None)
    accuracy = accuracy_score(y_test, y_random)
    print(f"\nBaseline accuracy: {accuracy}")

    y_test_predicted = clf.predict(x_test_norm)
    accuracy = accuracy_score(y_test, y_test_predicted)
    print(f"\nTesting set accuracy: {accuracy}")

    # import pdb; pdb.set_trace()

    y_test_predictions = []

    action_video = ActionVideo(22, 'boxing', 'd1')
    for i, hu_set in enumerate(action_video.frame_hu_set_generator()):
        hu_set[~np.isfinite(hu_set).any(axis=1)] = np.zeros(NUM_HU)

        # print(hu_set)
        hu_set_norm = normalize(hu_set, norm="l2")

        action_pred = clf.predict(hu_set).astype(np.uint8)

        counts = np.bincount(action_pred)
        prediction =  np.argmax(counts)
        if i == 10:
            import pdb; pdb.set_trace()
        print(prediction, clf.predict(x_test_norm[i].reshape(1, -1)))
        y_test_predictions.append(prediction)


    if show_graph:
        cm = confusion_matrix(y_test, y_test_predicted)

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

        title = "Confusion Matrix - Testing Data"

        disp = plot_confusion_matrix(
            clf,
            x_test_norm,
            y_test,
            display_labels=class_names,
            cmap=plt.cm.Reds,
            normalize="true",
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

        plt.show()

        disp.figure_.savefig("confusion_matrix_testing.png")
#
#     # Plot normalized confusion matrix
#     fig, ax = plot_confusion_matrix(
#         y_test.astype(np.uint8),
#         y_test_predicted.astype(np.uint8),
#         classes=labels,
#         normalize=True,
#     )
#
#     fig.savefig("confusion_matrix.png")
#
#     plt.show()
