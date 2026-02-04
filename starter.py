import asyncio
import uuid
from temporalio.client import Client

from shared import InputVideo
import sys

async def main():
    if len(sys.argv) < 2:
        print("Usage: python starter.py <filepath>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    client = await Client.connect("localhost:7233")
    result = await client.execute_workflow(
        "VideoProcessingWorkflow",
        InputVideo(filepath=filepath),
        id=f"video-processing-workflow-{uuid.uuid4()}",
        task_queue="video-processing-queue",
    )
    print("Workflow result:", result)

if __name__ == "__main__":
    asyncio.run(main())