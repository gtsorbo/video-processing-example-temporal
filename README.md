# Temporal Video Analysis Workflow example

## Overview
This repo demonstrates a Temporal workflow that:
1) detects shot boundaries in a video,
2) splits the video into shot clips, and
3) fan-outs analysis of each clip through a local FastAPI service.

### Contents
- [`workflows.py`](./workflows.py): contains the Temporal workflow `VideoProcessingWorkflow`
- [`activities.py`](./activities.py): contains the Temporal activites `detect_shots`, `split_shots`, and `analyze_shot`
- [`worker.py`](./worker.py): contains the Temporal worker configuraiton responsible for the above workflows and activites
- [`starter.py`](./starter.py): contains a starter function to execute the `VideoProcessingWorkflow` with a provided video file
- [`analysis_server.py`](./analysis_server.py): a local video analysis server providing descriptive keywords for provided video clips via FastAPI

## Prerequisites
- [Python 3.10+](https://www.python.org)
- [Temporal CLI](https://docs.temporal.io/cli)
- [FFmpeg](https://www.ffmpeg.org/download.html) (used by `scenedetect` to split clips)
> [!TIP]
> Mac:
> ```bash
> brew install temporal
> brew install ffmpeg
> ```

## Quickstart
1. Clone this repo and open the resulting directory
    ```bash
    git clone https://github.com/gtsorbo/video-processing-example-temporal.git
    cd video-processing-example-temporal
    ```
1. Create and activate a virtual environment, then install dependencies:
    ```bash
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```
1. Set the environment variable for your video filepath:
    ```bash
    export $VIDEOPATH=/path/to/video.mp4
    ```
1. In a new terminal, start the analysis API:
    ```bash
    uvicorn analysis_server:app --host 127.0.0.1 --port 8008
    ```
1. In another new terminal, start the Temporal server:
    ```bash
    temporal server start-dev
    ```
1. In another new terminal, start the Temporal worker:
    ```bash
    python worker.py
    ```
1. In another new terminal, execute the workflow with a local video path:
    ```bash
    python starter.py $VIDEOPATH
    ```