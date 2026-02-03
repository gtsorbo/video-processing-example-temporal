from datetime import timedelta
import asyncio

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities import detect_shots, split_shots, analyze_shot
    from shared import InputVideo, SplitterConfig

@workflow.defn
class VideoProcessingWorkflow:
    @workflow.run
    async def run(self, video: InputVideo) -> str:
        retry_policy = RetryPolicy(
            maximum_attempts=3,
            maximum_interval=timedelta(seconds=10),
        )

        analyze_retry = RetryPolicy(
            maximum_attempts=6,
            maximum_interval=timedelta(seconds=15),
            non_retryable_error_types=["NonRetryable"],
        )

        # Detect shots
        scene_list = await workflow.execute_activity(
            detect_shots,
            video,
            schedule_to_close_timeout=timedelta(seconds=60),
        )

        # Split shots
        result_code, scene_paths = await workflow.execute_activity(
            split_shots,
            SplitterConfig(
                video=video,
                scene_list=scene_list,
            ),
            schedule_to_start_timeout=timedelta(seconds=10),
            schedule_to_close_timeout=timedelta(seconds=300)
        )

        # Fan-out: start all analyze_shot activities without awaiting them yet
        analyze_shots = [
            workflow.execute_activity(
                analyze_shot,
                scene,
                schedule_to_start_timeout=timedelta(seconds=600),
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=analyze_retry,
            )
            for scene in scene_paths
        ]

        # Fan-in: wait for all of them to finish
        # return_exceptions=True will allow all to complete, even if some fail
        analysis_results = await asyncio.gather(*analyze_shots, return_exceptions=True)

        return (
            f"Video processing completed with result code: {result_code}. "
            f"Scenes: {len(scene_paths)}. "
            f"Analysis results: {analysis_results}"
        )
