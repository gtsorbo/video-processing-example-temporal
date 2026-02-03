from dataclasses import dataclass
from scenedetect import SceneList
from scenedetect import FrameTimecode


@dataclass
class InputVideo:
    filepath: str

@dataclass
class SplitterConfig:
    video: InputVideo
    scene_list: list

def scene_list_to_jsonable(scene_list: SceneList) -> list:
    out = []
    for start, end in scene_list:
        out.append({
            "start_frame": int(start.get_frames()),
            "end_frame": int(end.get_frames()),
            "fps": float(start.get_framerate()),
        })
    return out

def jsonable_to_scene_list(data) -> SceneList:
    scene_list = list()
    for item in data:
        start = item["start_frame"]
        end = item["end_frame"]
        fps = item["fps"]
        scene_list.append(
            tuple(
                [FrameTimecode(start, fps),
                 FrameTimecode(end, fps)]
            )
        )
    return scene_list