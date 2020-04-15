from config import training_sequence, test_sequence
from core import ActionVideo
from config import actions, backgrounds, frame_sequences
import numpy as np
from core import mp4_video_writer, OUTPUT_DIR, VID_DIR
import cv2

def get_action_video_objects():
    action_videos = []

    total = 0

    len_final_video = 2.0

    sequence = np.copy(test_sequence)
    np.random.seed(83)
    np.random.shuffle(sequence)

    for num in sequence:
        for action in actions:
            background = 'd1'
            key_name = f"person{num:02d}_{action}_{background}"

            range_indx = np.random.choice(
                list(range(len(frame_sequences[key_name]))),
                replace=True
            )

            frame_range = frame_sequences[key_name][range_indx]

            print(f'{key_name}: {frame_sequences[key_name][range_indx]}')

            total += frame_range[1] - frame_range[0] + 1

            cumulative_time = total / 25 / 60

            if cumulative_time <= len_final_video:
                print(f'Total: {total}: Time: {cumulative_time}')
                # pass
                av = ActionVideo(num, action, background)
                action_videos.append((av, frame_range))
            else:
                print("skipping")

    return action_videos


if __name__ == '__main__':

    action_video_tuples = get_action_video_objects()

    filename = 'spliced_action_video'

    out_path = f"{VID_DIR}/{filename}.mp4"

    fps = 25

    h, w, _ = action_video_tuples[0][0].video_frame_array.shape

    video_out = mp4_video_writer(out_path, (w, h), fps)

    for av, frame_range in action_video_tuples:
        print(av, frame_range)

        start = frame_range[0] - 1
        end = frame_range[1]

        print('Writing video ...')

        for i in range(start, end):

            output_frame = av.video_frame_array[:, :, i]

            out_frame = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)

            video_out.write(out_frame)

    video_out.release()