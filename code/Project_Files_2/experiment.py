import pprint
# import warnings

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler, normalize

import config
import core

matplotlib.use("Qt5Agg")



# warnings.filterwarnings("ignore")


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


def plot_confusion_matrix(
    y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()

    return fig, ax


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=200)

    # run_moment_calculation()

    X_train, y_train = core.generate_data(config.training_sequence)
    # X_validation, y_validation = core.generate_data(config.validation_sequence)
    X_test, y_test = core.generate_data(config.test_sequence)

    # Normalize the data
    x_train_norm = normalize(X_train, norm='l2')
    x_test_norm = normalize(X_test, norm='l2')

    clf = RandomForestClassifier(n_estimators=100)
    # clf = KNeighborsClassifier()
    clf.fit(x_train_norm, y_train)

########################################################################################
    # parameters = {'n_estimators': [50, 100, 150],
    #               'max_depth': [None, 10, 50, 100, 500, 1000]}
    # clf = GridSearchCV(clf, parameters, cv=10, refit=True)
    #
    # clf.fit(X_train, y_train)

    accuracy = clf.score(x_train_norm, y_train)
    print(f"Training set accuracy: {accuracy}")

    y_test_predicted = clf.predict(x_test_norm)
    accuracy = accuracy_score(y_test, y_test_predicted)
    print(f"Testing set accuracy: {accuracy}")

    import pdb; pdb.set_trace()

    print("Should you save this model?")



#     labels = np.array(
#         [
#             "blank",
#             "boxing",
#             "handclapping",
#             "handwaving",
#             "jogging",
#             "running",
#             "walking",
#         ]
#     )
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
# ########################################################################################
