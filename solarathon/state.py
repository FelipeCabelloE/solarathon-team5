import cv2
import numpy as np
import solara as sl
import pandas as pd

from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

class VideoProcessor:

    video_frame = sl.reactive(0)
    analysis_types = ['Pose', 'Detect']
    analysis_type = sl.reactive('')
    model = None

    raw_frames = []
    processed_frames = []
    processed_data = []
    processed_data_df = sl.reactive(pd.DataFrame())

    active_frame = sl.reactive(np.zeros((10, 10)).astype(int))

    set_frame_progress = sl.reactive(lambda: None)

    files_df = sl.reactive(pd.DataFrame([['Golf', 'golf.mp4', 30, True],
                                         ['Lifting', 'lifting.mp4', 10, True],
                                         ['Skateboarding', 'skateboarding.mp4', 10, True],
                                         ['Diving', 'diving.mp4', 10, True]],
                                         columns=['Sport', 'Name', 'FPS', 'Default example']))
    name = sl.reactive('')

    @classmethod
    def load_model(VideoProcessor, value):
        if value == 'Pose':
            VideoProcessor.model = YOLO('yolov8s-pose.pt')
        elif value == 'Detect':
            VideoProcessor.model = YOLO('yolov8s.pt')
        else:
            return None
    
    @classmethod
    def load_video(VideoProcessor, filename):
        video = cv2.VideoCapture(filename)
        frames = []
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames

        for frame_index in range(frame_count):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # Seek to the specific frame
            ret, frame = video.read()  # Read the frame
            if ret:
                frames.append(frame)
        VideoProcessor.raw_frames = frames

    @classmethod
    def process_video(VideoProcessor):
        if VideoProcessor.analysis_type.value == 'Pose':
            processed_data = []
            processed_frames = []
            complete_df = pd.DataFrame()
            for frame_idx, frame in enumerate(VideoProcessor.raw_frames):
                VideoProcessor.set_frame_progress(np.round(100 * (1+frame_idx)/len(VideoProcessor.raw_frames)))
                # Interestingly, if the frames are saved in BGR it works fine
                # but if we convert to RGB on loading, it crashes at frame 95
                results = VideoProcessor.model(Image.fromarray(frame))

                if results is not None:
                    for result in results:
                        kpoints = result.keypoints
                        sample_df = pd.DataFrame(np.squeeze(np.array(kpoints.xy)), columns=['coord0', 'coord1'])
                        sample_df['frame'] = frame_idx
                        sample_df['keypoint'] = np.arange(np.squeeze(np.array(kpoints.xy)).shape[0])
                        complete_df = pd.concat([complete_df, sample_df], axis=0, ignore_index=True)
                        processed_data.append(np.array(kpoints.xy))
                        for (x, y) in np.squeeze(np.array(kpoints.xy)):
                            cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), -1)
                        processed_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            VideoProcessor.processed_data = processed_data
            VideoProcessor.processed_frames = processed_frames
            VideoProcessor.processed_data_df.value = complete_df
            VideoProcessor.active_frame.value = processed_frames[VideoProcessor.video_frame.value]

        elif VideoProcessor.analysis_type.value == 'Detect':
            processed_frames = []
            complete_df = pd.DataFrame()
            for frame_idx, frame in enumerate(VideoProcessor.raw_frames):
                VideoProcessor.set_frame_progress(np.round(100 * (1+frame_idx)/len(VideoProcessor.raw_frames)))
                # Interestingly, if the frames are saved in BGR it works fine
                # but if we convert to RGB on loading, it crashes at frame 95
                results = VideoProcessor.model(Image.fromarray(frame))

                if results is not None:
                    boxes = results[0].boxes.xyxy.tolist()
                    classes = results[0].boxes.cls.tolist()
                    names = results[0].names
                    confidences = results[0].boxes.conf.tolist()
                    annotator = Annotator(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), line_width=2, example=str(names))
                    sample_df = pd.DataFrame()
                    sample_df['classes'] = [names[i] for i in classes]
                    sample_df['confidences'] = confidences
                    sample_df['frame'] = frame_idx
                    # In case there are repeated objects - assumes only one person per frame
                    sample_df = sample_df.groupby('classes').mean().reset_index()
                    complete_df = pd.concat([complete_df, sample_df], axis=0, ignore_index=True)

                    # Iterate through the results
                    for box, clse, conf in zip(boxes, classes, confidences):
                        annotator.box_label(box, names[int(clse)], (255, 42, 4))
                    processed_frames.append(annotator.result())

            VideoProcessor.processed_frames = processed_frames
            VideoProcessor.processed_data_df.value = complete_df
            VideoProcessor.active_frame.value = processed_frames[VideoProcessor.video_frame.value]

    @classmethod
    def update_frame(VideoProcessor, value):
        VideoProcessor.active_frame.value = VideoProcessor.processed_frames[value]
