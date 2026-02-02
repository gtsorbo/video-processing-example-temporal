from temporalio import activity
from scenedetect import detect, ContentDetector, split_video_ffmpeg, SceneList

from shared import InputVideo


@activity.defn
async def detect_shots(video: InputVideo) -> SceneList:
    scene_list = detect(video.filepath, ContentDetector())
    return scene_list

@activity.defn
async def split_shots(video: InputVideo, scene_list: any) -> int:
    code = split_video_ffmpeg(video.filepath, scene_list)
    return code
