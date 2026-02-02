from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities import detect_shots, split_shots
    from shared import InputVideo


@workflow.defn
class VideoSceneSplittingWorkflow:
    @workflow.run
    async def run(self, video: InputVideo) -> str:
        retry_policy = RetryPolicy(
            maximum_attempts=3,
            maximum_interval=timedelta(seconds=2),
        )

        # Detect shots
        scene_list = await workflow.execute_activity(
            detect_shots,
            video,
            schedule_to_close_timeout=timedelta(seconds=10),
        )

        # Split shots
        result_code = await workflow.execute_activity(
            split_shots,
            video,
            scene_list,
            schedule_to_close_timeout=timedelta(seconds=30),
        )

        return f"Video processing completed with result code: {result_code}"
