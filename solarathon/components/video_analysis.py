import cv2
import numpy as np
import solara as sl
import pandas as pd

from PIL import Image
from ultralytics import YOLO

class VideoProcessor:

    video_frame = sl.reactive(0)
    analysis_types = ['Pose', 'Track', 'Detect']
    analysis_type = sl.reactive('')
    model = None
    name = sl.reactive("")

    raw_frames = []
    processed_frames = []
    processed_data = []

    active_frame = sl.reactive(np.zeros((10, 10)).astype(int))

    set_frame_progress = sl.reactive(lambda: None)

    files_df = sl.reactive(pd.DataFrame(
        [['golf.mp4']], columns=['Name']))     

    @classmethod
    def load_model(VideoProcessor, value):
        if value == 'Pose':
            VideoProcessor.model = YOLO('yolov8n-pose.pt')
        elif value == 'Detect':
            VideoProcessor.model = YOLO('yolov8s.pt')
        else:
            return None    

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
            VideoProcessor.processed_data_df = complete_df
            VideoProcessor.active_frame.value = processed_frames[VideoProcessor.video_frame.value]

    @classmethod
    def update_frame(VideoProcessor, value):
        VideoProcessor.active_frame.value = VideoProcessor.processed_frames[value]