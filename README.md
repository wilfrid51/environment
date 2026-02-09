# Trace Task Evaluation Environment

A Python-based evaluation framework for testing Large Language Models (LLMs) on trace tasks. The system generates Python code challenges with injected debug print statements and evaluates LLM predictions by comparing their expected stdout output with the actual execution results.

## Overview

This codebase provides:

1. **Trace Task Generation**: Creates Python code challenges with deterministic print statement injections
2. **LLM Evaluation**: Tests LLM models on predicting the exact stdout output of Python programs
3. **OpenEnv Training Interface**: Supports reinforcement learning training with reset/step/state/stop methods
4. **Flexible LLM Integration**: Works with any OpenAI-compatible API endpoint

## Project Structure

```
Env/
├── core/                   # Core utilities
│   ├── llm_chat.py         # LLM chat completion helper with streaming support
│   └── openenv.py          # OpenEnv protocol models for training interface
├── trace/                  # Trace task implementation
│   ├── __init__.py         # Package initialization
│   ├── env.py              # Main Actor class with evaluation and training interfaces
│   ├── trace_task.py       # Task generator and evaluator with print injection
│   ├── models.py           # Data models (Challenge)
│   └── request_logger.py   # Structured logging utility
├── evaluate.py             # Main evaluation script
├── inference.sh            # vLLM server startup script
└── requirement.txt         # Python dependencies

```

## Features

### 1. Print Injection System
- Injects deterministic debug print statements into Python code
- Uses AST transformation to add execution-dependent prints
- Prevents overfitting by using variable values rather than memory addresses
- Configurable number of injections per program

### 2. Dual Interface Modes
- **Evaluation Mode**: One-shot LLM evaluation with internal generation and scoring
- **Training Mode**: OpenEnv-compatible interface for external control (reset/step/state/stop)

### 3. LLM Integration
- Streaming and non-streaming support
- Automatic retry with exponential backoff
- Per-chunk timeout protection
- Support for reasoning models (o1-style)
- Think-tag stripping

### 4. Flexible Evaluation
- Normalized output comparison (whitespace and case insensitive)
- Handles markdown code blocks in LLM responses
- Removes reasoning/thinking tags automatically

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirement.txt
   ```

2. **Set up environment variables:**
   ```bash
   export LLM_BASE_URL="http://localhost:20000/v1"
   export LLM_API_KEY="your_api_key"
   export LLM_MODEL="Qwen/Qwen3-4B-Instruct-2507"
   export HF_TOKEN="your_huggingface_token"  # Optional, for dataset access
   ```

3. **Start vLLM server (optional, for local inference):**
   ```bash
   bash inference.sh
   ```

## Usage

### Basic Evaluation

Run evaluation on a set of tasks:

```python
python evaluate.py
```

The script will:
- Generate 100 random task IDs
- Evaluate each task with the configured LLM
- Print results and calculate average accuracy

### Programmatic Usage

#### Evaluation Mode

```python
import asyncio
import trace

async def main():
    actor = trace.Actor()
    
    result = await actor.evaluate(
        task_id=12345,
        seed=42,
        api_key="your_api_key",
        model="Qwen/Qwen3-4B-Instruct-2507",
        base_url="http://localhost:20000/v1",
    )
    
    print(f"Score: {result['score']}")
    print(f"Success: {result['success']}")

asyncio.run(main())
```

#### Training Mode (OpenEnv Interface)

```python
import asyncio
import trace

async def main():
    actor = trace.Actor(api_key="your_api_key")
    
    # Reset to start a new episode
    response = await actor.reset(task_id=12345, seed=42)
    episode_id = response.episode_id
    observation = response.observation  # Challenge prompt
    
    # Generate prediction (using your LLM)
    prediction = "your_predicted_stdout_output"
    
    # Step with the prediction
    result = await actor.step(action=prediction, episode_id=episode_id)
    print(f"Reward: {result.reward}")
    print(f"Done: {result.done}")
    
    # Clean up
    await actor.stop(episode_id=episode_id)

asyncio.run(main())
```

## Configuration

### Environment Variables

- `LLM_BASE_URL`: Base URL for LLM API (default: `http://localhost:20000/v1`)
- `LLM_API_KEY`: API key for LLM service
- `LLM_MODEL`: Model name to use (default: `Qwen/Qwen3-4B-Instruct-2507`)
- `HF_TOKEN`: HuggingFace token for dataset access (optional)
- `TRACE_LOG_LEVEL`: Logging level (default: `INFO`)

### Dataset

The system uses the `satpalsr/rl-python` dataset by default. You can modify the dataset in `trace/trace_task.py`:

```python
TraceTask(
    dataset_name="your_dataset",
    dataset_split="train",
    dataset_shuffle=False,
)
```

## How It Works

1. **Task Generation**:
   - Loads a Python program from the dataset
   - Injects debug print statements using AST transformation
   - Executes the transformed code to get ground truth stdout
   - Creates a challenge prompt asking the LLM to predict the output

2. **LLM Evaluation**:
   - Sends the challenge prompt to the LLM
   - Receives the predicted stdout output
   - Cleans the response (removes markdown, thinking tags)
   - Compares with ground truth using normalized comparison

3. **Scoring**:
   - Binary score: 1.0 if outputs match (normalized), 0.0 otherwise
   - Normalization handles whitespace differences and case variations

## Development

### Adding Comments

All files include comprehensive comments explaining:
- Module purpose and functionality
- Function parameters and return values
- Complex logic and algorithms
- Usage examples where helpful

### Debugging

Enable HTTP request logging:

```python
import debug_requests
# Then run your evaluation script
```

This will show detailed HTTP request/response information for LLM API calls.

## Dependencies

- `vllm`: For local LLM inference server
- `datasets`: For loading HuggingFace datasets
- `openai`: OpenAI-compatible API client
- `httpx`: HTTP client library
- `pydantic`: Data validation
- `structlog`: Structured logging

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
