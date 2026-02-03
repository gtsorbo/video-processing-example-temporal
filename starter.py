import asyncio
import uuid
from temporalio.client import Client

from shared import InputVideo

async def main():
    client = await Client.connect("localhost:7233")
    result = await client.execute_workflow(
        "VideoProcessingWorkflow",
        InputVideo(filepath="/Users/Grant/Desktop/VideoTest/TDWP2.mp4"),
        id=f"video-processing-workflow-{uuid.uuid4()}",
        task_queue="video-processing-queue",
    )
    print("Workflow result:", result)

if __name__ == "__main__":
    asyncio.run(main())