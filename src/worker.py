import asyncio
from temporalio.client import Client
from temporalio import workflow
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from workflows import SayHelloWorkflow
    from src.activities import greet

async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue="my-task-queue",
        workflows=[SayHelloWorkflow],
        activities=[greet],
    )
    print("Worker started.")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
