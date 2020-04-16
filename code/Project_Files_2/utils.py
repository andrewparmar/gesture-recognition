import cv2
import matplotlib.pyplot as plt


def plot_features(features, **kwargs):
    labels = ["h1", "h2", "h3", "h4", "h5", "h6", "h7"]

    fig, ax = plt.subplots()

    ax.plot(features[:, :7])
    ax.legend(labels, bbox_to_anchor=(1, 1))
    ax.set(
        xlabel="Frame number", ylabel="Hu Value", title=f'{kwargs.get("title", "Plot")}'
    )

    plt.show()


def mp4_video_writer(filename, frame_size, fps=25):
    """Opens and returns a video for writing.
    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.
    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def print_fraction(i, total):
    fraction = (i + 1) / total
    if fraction < 1:
        print('\r', 'Progress [{:>7.2%}]'.format(fraction), end='')
    else:
        print('\r', 'Progress [{:>7.2%}]'.format(fraction))


def add_text_to_img(img, text, coordinate):
    x, y = coordinate
    font = cv2.FONT_ITALIC
    fontScale = 0.7
    color_outline = (255, 255, 255)
    thickness_outline = 2
    org = (x, y)

    cv2.putText(img, text, org, font, fontScale, color_outline, thickness_outline)

    return img


def gray_frame_generator(file_path):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        file_path (string): Relative file path.

    Returns:
        None.
    """
    video = cv2.VideoCapture(file_path)

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield gray_image
        else:
            break

    video.release()
    yield None


def get_video_frame_count(file_path):
    video = cv2.VideoCapture(file_path)

    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    video.release()

    return total
