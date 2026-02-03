import asyncio
from temporalio.client import Client
from temporalio import workflow
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from workflows import VideoProcessingWorkflow
    from activities import detect_shots, split_shots, analyze_shot

async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue="video-processing-queue",
        workflows=[VideoProcessingWorkflow],
        activities=[detect_shots, split_shots, analyze_shot],
        max_concurrent_activities=3
    )
    print("Worker started.")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
