import os
import asyncio
import random
import trace

def evaluate_task_id():
    task_list = []

    for i in range(100):
        task_id = random.randint(0, 100000000)
        task_list.append(task_id)

    return task_list

async def evaluate(task_id):
    actor = trace.Actor()

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:20000/v1")
    api_key = os.getenv("LLM_API_KEY", "your_api_key")
    model = os.getenv("LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

    return await actor.evaluate(
        task_id=task_id,
        seed=random.randint(0, 100000000),
        api_key=api_key,
        model=model,
        base_url=base_url,
    )

async def run():
    tasks = evaluate_task_id()

    max_concurrent = 1
    accuracy = 0.0

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_task(task_id):
        nonlocal accuracy
        async with semaphore:
            result = await evaluate(task_id)
            if result is None:
                print("result is None")
            print(result)
            print(f"# Task: {result['extra']['task_id']} \tScore: {result['score']}")
            accuracy += result['score']

    pending_tasks = [process_task(task_id) for task_id in tasks]
    results = await asyncio.gather(*pending_tasks)

    print(f"Average Accuracy: {accuracy / len(tasks)}")

if __name__ == "__main__":
    asyncio.run(run())