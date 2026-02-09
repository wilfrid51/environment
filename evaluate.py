"""
Main evaluation script for trace task evaluation.

This script runs batch evaluation on multiple trace tasks, testing LLM models
on their ability to predict the exact stdout output of Python programs with
injected debug print statements.

Usage:
    python evaluate.py

The script will:
1. Generate a list of random task IDs
2. Evaluate each task using the configured LLM
3. Print results and calculate average accuracy

Environment variables:
    LLM_BASE_URL: Base URL for LLM API (default: http://localhost:20000/v1)
    LLM_API_KEY: API key for LLM service
    LLM_MODEL: Model name to use (default: Qwen/Qwen3-4B-Instruct-2507)
"""

import os
import asyncio
import random
import trace


def evaluate_task_id():
    """
    Generate a list of random task IDs for evaluation.
    
    Returns:
        List[int]: List of 100 random task IDs in the range [0, 100000000)
    """
    task_list = []

    for i in range(100):
        task_id = random.randint(0, 100000000)
        task_list.append(task_id)

    return task_list


async def evaluate(task_id):
    """
    Evaluate a single trace task using the configured LLM.
    
    Args:
        task_id: Task identifier (used as index into dataset)
    
    Returns:
        dict: Evaluation result containing:
            - score: Binary score (1.0 for correct, 0.0 for incorrect)
            - success: Boolean indicating if score > 0
            - time_taken: Time taken for evaluation
            - extra: Additional metadata (conversation, seed, test_result, etc.)
    """
    # Initialize the trace actor
    actor = trace.Actor()

    # Get configuration from environment variables
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:20000/v1")
    api_key = os.getenv("LLM_API_KEY", "your_api_key")
    model = os.getenv("LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

    # Run evaluation with random seed for reproducibility
    return await actor.evaluate(
        task_id=task_id,
        seed=random.randint(0, 100000000),
        api_key=api_key,
        model=model,
        base_url=base_url,
    )


async def run():
    """
    Main evaluation loop.
    
    Generates task IDs, evaluates them concurrently (with configurable concurrency),
    and prints results with average accuracy.
    """
    # Generate list of task IDs to evaluate
    tasks = evaluate_task_id()

    # Configuration: max concurrent evaluations (set to 1 for sequential execution)
    max_concurrent = 40
    accuracy = 0.0

    # Semaphore to limit concurrent evaluations
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_task(task_id):
        """
        Process a single task evaluation.
        
        Args:
            task_id: Task identifier to evaluate
        """
        nonlocal accuracy
        # Acquire semaphore to limit concurrency
        async with semaphore:
            result = await evaluate(task_id)
            if result is None:
                print("result is None")
            print(result)
            # Print formatted result with task ID and score
            print(f"# Task: {result['extra']['task_id']} \tScore: {result['score']}")
            # Accumulate accuracy for average calculation
            accuracy += result['score']

    # Create tasks for all evaluations
    pending_tasks = [process_task(task_id) for task_id in tasks]
    # Execute all tasks concurrently (limited by semaphore)
    results = await asyncio.gather(*pending_tasks)

    # Print final average accuracy
    print(f"Average Accuracy: {accuracy / len(tasks)}")


if __name__ == "__main__":
    # Run the async evaluation loop
    asyncio.run(run())