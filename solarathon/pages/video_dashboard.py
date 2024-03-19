import cv2
import os
import tempfile
import time

from typing import Any, Dict, Optional, cast

import ipywidgets
import numpy as np
import pandas as pd
import plotly.express as px
import solara as sl

from matplotlib.figure import Figure
from PIL import Image
from solara.components.file_drop import FileInfo

from solarathon.state import VideoProcessor
from solarathon.pages import SharedComponent
from solara.alias import rv

@sl.component
def FrameViewer():
    sl.Image(VideoProcessor.active_frame.value, width="500px") 

@sl.component
def FrameVideo():
    url = f'static/public/{VideoProcessor.name.value}'
    ipywidgets.Video.element(value=url.encode('utf8'), format='url', width=500)

@sl.component
def AnalysisViewer():

    selection_data, set_selection_data = sl.use_state(None)
    click_data, set_click_data = sl.use_state(None)
    hover_data, set_hover_data = sl.use_state(None)
    unhover_data, set_unhover_data = sl.use_state(None)
    deselect_data, set_deselect_data = sl.use_state(None)

    mask_frame = VideoProcessor.processed_data_df.value["frame"] < VideoProcessor.video_frame.value
    
    if VideoProcessor.analysis_type.value == 'Pose':
        mask_keypoints_nonzero = (VideoProcessor.processed_data_df.value['coord0'] > 0) | (VideoProcessor.processed_data_df.value['coord1'] > 0) 
        fig=px.line(VideoProcessor.processed_data_df.value[mask_frame & mask_keypoints_nonzero], x='coord0', y='coord1', color='keypoint')
        fig.update_xaxes(range=[0, VideoProcessor.active_frame.value.shape[1]])
        fig.update_yaxes(range=[0, VideoProcessor.active_frame.value.shape[0]])
    elif VideoProcessor.analysis_type.value == 'Detect':
        fig=px.line(VideoProcessor.processed_data_df.value[mask_frame], x='frame', y='confidences', color='classes')
        fig.update_xaxes(range=[0, len(VideoProcessor.raw_frames)])
        fig.update_yaxes(range=[-0.05, 1.05])

    fig.update_layout(width=600, height=600)

    sl.FigurePlotly(
        fig, on_selection=set_selection_data, on_click=set_click_data, on_hover=set_hover_data, on_unhover=set_unhover_data, on_deselect=set_deselect_data,
    )
    return


@sl.component
def Page():

    file_status, set_file_status = sl.use_state('No file received.')
    analysis_status, set_analysis_status = sl.use_state('Analysis has not started. Please select or upload a video and press "Start analysis"!')
    frame_progress, set_frame_progress = sl.use_state(0.0)
    VideoProcessor.set_frame_progress = set_frame_progress
    show_video_player, set_show_video_player = sl.use_state(True)
    process_video_react = sl.use_reactive(False)
    analysis_complete = sl.use_reactive(False)
    sport_name = sl.reactive('')
    sport_clip_fps = sl.reactive(0)
    start_analysis, set_start_analysis = sl.use_state(True)

    # Select among default videos:
    def on_action_cell(column, row_index):
        selected_video_name = VideoProcessor.files_df.value.loc[row_index, 'Name']
        if bool(VideoProcessor.files_df.value.loc[row_index, 'Default example']):
            set_file_status(f'New file: {selected_video_name}')
            VideoProcessor.name.value = selected_video_name
            VideoProcessor.load_video(f'/workspaces/solarathon-team5/solarathon/public/{selected_video_name}')
    cell_actions = [sl.CellAction(name="Load video", on_click=on_action_cell)]

    def on_file(file: FileInfo):
        set_file_status(f'New file: {file["name"]}')
        VideoProcessor.name.value = file["name"]

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(file["data"])
        temp_file.close()
        VideoProcessor.temp_filename = temp_file.name

        # VideoProcessor.raw_frames = frames
        VideoProcessor.load_video(VideoProcessor.temp_filename)

        # Update DF with template videos
        df = VideoProcessor.files_df.value.copy()
        df.loc[len(VideoProcessor.files_df.value)] = [file["name"], False]
        VideoProcessor.files_df.value = df

    def clear_files():
        # Remove temp file from the video
        tempfile.TemporaryDirectory().cleanup()
        #if os.path.isfile(VideoProcessor.temp_filename):
        #    os.remove(VideoProcessor.temp_filename)

    def process_video():
        if process_video_react.value:
            VideoProcessor.process_video()
            analysis_complete.set(True)
            process_video_react.set(False)

    sl.use_thread(process_video, dependencies=[process_video_react.value])

    # Interface
    with sl.Column() as main:
        sl.Title("Video dashboard")
        with sl.Sidebar():
            SharedComponent()

        with sl.ColumnsResponsive(small=12, large=[4, 8]):
            with sl.Card(sl.Text(text=""), style={"max-width": "500px"}, margin=0, classes=["my-2"]):
                sl.Markdown('### Video files:')
                sl.DataFrame(VideoProcessor.files_df.value, items_per_page=5, cell_actions=cell_actions)
                sl.Markdown('### Video upload:')
                sl.Info(file_status)
                with sl.GridFixed(columns=2):
                    sl.InputText('Sport name:', value=sport_name)
                    sl.InputInt('Video FPS:', value=sport_clip_fps)
                sl.FileDrop(label='Alternatively, please provide a video to analyse.', lazy=False, on_file=on_file)
                
                def change_analysis_type(value):
                    analysis_complete.value = False
                    VideoProcessor.set_frame_progress(0)
                    VideoProcessor.load_model(value) 

                sl.Select(label='Type of analysis', values=VideoProcessor.analysis_types,
                        value=VideoProcessor.analysis_type, on_value=change_analysis_type)
                if (len(VideoProcessor.analysis_type.value) > 1) and (len(VideoProcessor.name.value) > 1):
                    set_start_analysis(False)
                with sl.Column():
                    sl.ProgressLinear(value=frame_progress, color="blue")

                    def start_process_video():
                        VideoProcessor.set_frame_progress(0)
                        process_video_react.set(True)

                    sl.Button(label='Start analysis', on_click=start_process_video, disabled=start_analysis)

                    if frame_progress == 0:
                        sl.Warning(label=analysis_status)
                    elif (frame_progress > 0) and (frame_progress < 99.5):
                        set_analysis_status(f'Running analysis: {int(frame_progress)} %')
                        sl.Info(label=analysis_status)
                    elif analysis_complete.value:
                        set_analysis_status('Analysis complete!')
                        sl.Success(label=analysis_status)
                    sl.Button(label='Clear temporary files', on_click=clear_files)

            if analysis_complete.value:
                if show_video_player:
                    with sl.Column():
                        FrameVideo()
                        sl.Switch(label="Media player", value=show_video_player, on_value=set_show_video_player)
                else:
                    with sl.ColumnsResponsive(small=12, large=[8, 4]):
                        with sl.Column():
                            FrameViewer()
                            sl.SliderInt(label='Frame:', min=0, max=len(VideoProcessor.raw_frames)-1,
                                        value=VideoProcessor.video_frame, on_value=VideoProcessor.update_frame)                               
                            sl.Switch(label="Media player", value=show_video_player, on_value=set_show_video_player)

                        with sl.Column():
                            AnalysisViewer()
                    
    return main
