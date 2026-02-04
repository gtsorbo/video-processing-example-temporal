import os
from temporalio import activity
from temporalio.exceptions import ApplicationError

from scenedetect import detect, ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg, default_formatter, VideoMetadata, SceneMetadata
from pathlib import Path
import json
import httpx

from shared import InputVideo, SplitterConfig, scene_list_to_jsonable, jsonable_to_scene_list

@activity.defn
async def detect_shots(video: InputVideo) -> list:
    
    # Detect shot changes in the video using ContentDetector
    scene_list = detect(video.filepath, ContentDetector())
    return scene_list_to_jsonable(scene_list)

@activity.defn
async def split_shots(input: SplitterConfig) -> tuple[int, list]:
    
    video = input.video
    scene_list_json = input.scene_list

    stem = Path(video.filepath).stem
    output_dir = Path(video.filepath).parent / stem
    output_file_template: str = "$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scene_list = jsonable_to_scene_list(scene_list_json)

    # Prepare output directory and filename formatter
    base_fmt = default_formatter(output_file_template)
    output_paths = []
    
    # Custom formatter callback to capture output paths
    def capture_formatter(video: VideoMetadata, scene: SceneMetadata):
        filename = base_fmt(video, scene)
        output_paths.append(str((output_dir / filename)))
        return filename
    
    # Split the video into shots using the detected scene list
    code = split_video_ffmpeg(
        video.filepath, 
        scene_list, 
        output_dir=output_dir,
        formatter=capture_formatter, 
        show_output=True, 
        show_progress=True
    )
    return code, output_paths

@activity.defn
async def analyze_shot(scene_path: str) -> str:
    """
    Calls local analysis service to analyze the full clip.
    Returns JSON up to 10 keywords describing the clip.
    """
    clip_uri = Path(scene_path).as_uri()

    if not clip_uri or not clip_uri.startswith("file://"):
        raise ApplicationError("clip_uri must be file://...", type="NonRetryable")

    base_url = os.getenv("ANALYSIS_API_URL", "http://127.0.0.1:8008")
    timeout_s = float(os.getenv("ANALYSIS_API_TIMEOUT_S", "5.0"))

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(f"{base_url}/analyze/clip", json={"clip_uri": clip_uri})
    except httpx.TimeoutException as e:
        raise ApplicationError("Analysis API timeout", type="Transient") from e
    except httpx.RequestError as e:
        raise ApplicationError(f"Analysis API request error: {e}", type="Transient") from e

    if resp.status_code == 429:
        raise ApplicationError("Rate limited (429)", type="RateLimited")
    if 500 <= resp.status_code <= 599:
        raise ApplicationError(f"Server error ({resp.status_code})", type="Transient")
    if resp.status_code == 400:
        raise ApplicationError(f"Bad request: {resp.text}", type="NonRetryable")

    resp.raise_for_status()
    # normalize output to stable json string
    data = resp.json()

    # Write each clip analysis result to a local JSON file 
    with open(Path(scene_path).with_suffix('.json'), 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

