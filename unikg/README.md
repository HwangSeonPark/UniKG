# UniKG Port Configuration Guide

This document provides a detailed guide on how to configure port numbers in the UniKG project.

## Overview

Previously, port numbers were hardcoded in the source code, making it difficult to change them. The system has been improved to allow easy port configuration through environment variables.

## Changes Made

### 1. Removal of Hardcoded Port Numbers

Hardcoded port numbers have been replaced with environment variables in the following files:

- `verifier/extract.py`: vLLM server port configuration
- `verifier/run.py`: Refiner server port configuration
- `verifier/refiner.py`: Default port settings in the Refiner class

### 2. Environment Variable-Based Configuration

All port numbers are now configured through environment variables. Default values are provided in the `run.sh` script, but can be overridden by setting environment variables before running the script.

## Port Configuration Methods

### Method 1: Setting Environment Variables Before Running run.sh (Required)

The `verifier/run.sh` script requires port environment variables to be set before execution. The script will check for these variables and exit with an error if they are not set:

```bash
# vLLM server port settings (REQUIRED)
export VLLM_HOST="${VLLM_HOST:-localhost}"
export QWEN_PORT="<port_number>"
export MISTRAL_PORT="<port_number>"
export DEFAULT_VLLM_PORT="${DEFAULT_VLLM_PORT:-$QWEN_PORT}"

# Refiner server port settings (REQUIRED)
export REFINER_HOST="${REFINER_HOST:-localhost}"
export REFINER_PORT="<port_number>"
export REFINER_MAX_WORKERS="${REFINER_MAX_WORKERS:-10}"
export REFINER_MODEL="${REFINER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export REFINER_MAX_TOKENS="${REFINER_MAX_TOKENS:-10000}"

# Then run the script
bash verifier/run.sh qwen webnlg20
```

**Important**: The script will exit with an error if `QWEN_PORT`, `MISTRAL_PORT`, or `REFINER_PORT` are not set.

### Method 2: Using .env File (Optional)

You can create a `.env` file in the project root:

```bash
# .env file example
VLLM_HOST=localhost
QWEN_PORT=<port_number>
MISTRAL_PORT=<port_number>
DEFAULT_VLLM_PORT=<port_number>
REFINER_HOST=localhost
REFINER_PORT=<port_number>
REFINER_MAX_WORKERS=10
REFINER_MODEL=Qwen/Qwen2.5-7B-Instruct
REFINER_MAX_TOKENS=10000
```

Then load it before running the script:

```bash
export $(cat .env | xargs)
bash verifier/run.sh qwen webnlg20
```

## Environment Variables Reference

### vLLM Server Variables

| Variable Name | Description | Default Location | Used In |
|---------------|-------------|-------------------|---------|
| `VLLM_HOST` | vLLM server host address | `run.sh` | `extract.py` |
| `QWEN_PORT` | Qwen model server port | `run.sh` | `extract.py` |
| `MISTRAL_PORT` | Mistral model server port | `run.sh` | `extract.py` |
| `DEFAULT_VLLM_PORT` | Default vLLM port (used when model is not in map) | `run.sh` | `extract.py` |
| `VLLM_API_KEY` | vLLM API key | `run.sh` | `extract.py` |

### Refiner Server Variables

| Variable Name | Description | Default Location | Used In |
|---------------|-------------|-------------------|---------|
| `REFINER_HOST` | Refiner server host address | `run.sh` | `run.py`, `refiner.py` |
| `REFINER_PORT` | Refiner server port | `run.sh` | `run.py`, `refiner.py` |
| `REFINER_MAX_WORKERS` | Refiner maximum worker count | `run.sh` | `run.py`, `refiner.py` |
| `REFINER_MODEL` | Model name used by Refiner | `run.sh` | `run.py`, `refiner.py` |
| `REFINER_MAX_TOKENS` | Refiner maximum token count | `run.sh` | `run.py`, `refiner.py` |

## Code Changes Details

### extract.py Changes

**Before:**
```python
VLLM_HOST = "localhost"
VLLM_API_KEY = "none"
MODEL_PORT_MAP = {
    "Qwen/Qwen2.5-7B-Instruct": 3906,
    "mistralai/Mistral-7B-Instruct-v0.3": 8003
}
```

**After:**
```python
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "none")
QWEN_PORT = int(os.getenv("QWEN_PORT"))
MISTRAL_PORT = int(os.getenv("MISTRAL_PORT"))
MODEL_PORT_MAP = {
    "Qwen/Qwen2.5-7B-Instruct": QWEN_PORT,
    "mistralai/Mistral-7B-Instruct-v0.3": MISTRAL_PORT
}
```

### run.py Changes

**Before:**
```python
DEFAULT_PORT = 3906
DEFAULT_MAX_WORKERS = 10
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_TOKENS = 10000
```

**After:**
```python
DEFAULT_PORT = int(os.getenv("REFINER_PORT"))
DEFAULT_MAX_WORKERS = int(os.getenv("REFINER_MAX_WORKERS"))
DEFAULT_MODEL = os.getenv("REFINER_MODEL")
DEFAULT_MAX_TOKENS = int(os.getenv("REFINER_MAX_TOKENS"))
DEFAULT_HOST = os.getenv("REFINER_HOST", "localhost")
```

### refiner.py Changes

**Before:**
```python
def __init__(self, host="localhost", port=3906, max_workers=10, model="Qwen/Qwen2.5-7B-Instruct", max_tokens=10000):
```

**After:**
```python
def __init__(self, host=None, port=None, max_workers=10, model=None, max_tokens=10000):
    # Get values from environment variables if arguments are not provided
    if host is None:
        host = os.getenv("REFINER_HOST", "localhost")
    if port is None:
        port = int(os.getenv("REFINER_PORT"))
    if model is None:
        model = os.getenv("REFINER_MODEL")
```

## Usage Examples

### Example 1: Basic Usage with Required Ports

```bash
# Set required port environment variables
export QWEN_PORT=<port_number>
export MISTRAL_PORT=<port_number>
export REFINER_PORT=<port_number>

# Run the script
bash verifier/run.sh qwen webnlg20
```

### Example 2: Custom Port Configuration

```bash
# Set custom ports
export QWEN_PORT=5000
export MISTRAL_PORT=5001
export REFINER_PORT=5000

# Run with Qwen model
bash verifier/run.sh qwen webnlg20
```

### Example 3: Using Remote Server

```bash
# Configure for remote server
export VLLM_HOST="192.168.1.100"
export REFINER_HOST="192.168.1.100"
export QWEN_PORT=<port_number>
export MISTRAL_PORT=<port_number>
export REFINER_PORT=<port_number>

# Run the script
bash verifier/run.sh qwen webnlg20
```

### Example 4: Different Ports for Different Models

```bash
# Set ports for different models
export QWEN_PORT=<port_number>
export MISTRAL_PORT=<port_number>
export REFINER_PORT=<port_number>

# Run with Qwen model
bash verifier/run.sh qwen webnlg20

# Run with Mistral model (same ports can be reused)
bash verifier/run.sh mistral webnlg20
```

## Troubleshooting

### Port Conflict Error

If a port is already in use, change to a different port:

```bash
export QWEN_PORT=<different_port_number>
export MISTRAL_PORT=<different_port_number>
export REFINER_PORT=<different_port_number>
```

### Connection Failure Error

Verify that the server is running and check if the host and port are correct:

```bash
# Check local server
curl http://localhost:<port>/health

# Check remote server
curl http://<host>:<port>/health
```

### Environment Variables Not Set Error

If you see an error message about missing environment variables, make sure to set all required ports:

```bash
# Verify environment variables
echo $QWEN_PORT
echo $MISTRAL_PORT
echo $REFINER_PORT

# Set all required environment variables
export QWEN_PORT=<port_number>
export MISTRAL_PORT=<port_number>
export REFINER_PORT=<port_number>

# Then run the script
bash verifier/run.sh qwen webnlg20
```

## Important Notes

1. **Port numbers must be integers**: Environment variables are strings, but they are converted to integers using `int()` in the code.

2. **Required environment variables**: The following environment variables MUST be set before running `run.sh`:
   - `QWEN_PORT`: Port number for Qwen model server
   - `MISTRAL_PORT`: Port number for Mistral model server
   - `REFINER_PORT`: Port number for Refiner server
   
   If any of these are not set, the script will exit with an error message.

3. **Server must be running**: After setting ports, ensure that the server is running on the specified port.

4. **Firewall configuration**: When using remote servers, ensure that the firewall allows connections on the specified ports.

5. **No hardcoded ports**: Port numbers are no longer hardcoded in the source code. All ports must be configured via environment variables.

## Additional Information

- Required environment variables (`QWEN_PORT`, `MISTRAL_PORT`, `REFINER_PORT`) must be set before running `run.sh`.
- If you run Python scripts directly, you must set all required environment variables beforehand.
- The `run.sh` script validates that required port environment variables are set and provides helpful error messages if they are missing.

## Related Files

- `verifier/extract.py`: Uses vLLM server ports for triple extraction
- `verifier/run.py`: Uses Refiner server ports for triple refinement
- `verifier/refiner.py`: Refiner class implementation
- `verifier/run.sh`: Main execution script (includes port configuration)
