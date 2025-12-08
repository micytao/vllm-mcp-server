# Quick Start Guide

Get up and running with vLLM MCP Server in 5 minutes.

## Prerequisites

- **Podman** or **Docker** installed and running (Podman preferred)
- **Cursor** or **Claude Desktop** (or any MCP-compatible client)

## Step 1: Configure Your MCP Client

### For Cursor

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "vllm": {
      "command": "uvx",
      "args": ["vllm-mcp-server"],
      "env": {
        "VLLM_BASE_URL": "http://localhost:8000"
      }
    }
  }
}
```

### For Claude Desktop

Edit your Claude Desktop config:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "vllm": {
      "command": "uvx",
      "args": ["vllm-mcp-server"],
      "env": {
        "VLLM_BASE_URL": "http://localhost:8000"
      }
    }
  }
}
```

## Step 2: Restart Your MCP Client

Restart Cursor or Claude Desktop to load the new MCP server.

## Step 3: Start Using vLLM

Once configured, you can interact with vLLM through natural language. Here are some example prompts:

### Check Your Platform

> "Check my platform status for running vLLM"

This shows your platform, container runtime (Podman/Docker), and GPU availability.

### Start a vLLM Server

> "Start a vLLM server with TinyLlama/TinyLlama-1.1B-Chat-v1.0"

The server automatically detects your platform and selects the right container image.

### Monitor Loading Progress

> "Show me the vLLM container logs"

Watch the model download and loading progress.

### Check Server Health

> "Is the vLLM server ready?"

### Chat with the Model

> "Ask vLLM: What is the capital of France?"

Or for multi-turn conversations:

> "Start a chat with vLLM as a helpful coding assistant, then ask it to write a Python fibonacci function"

### Stop the Server

> "Stop the vLLM server"

---

## Quick Reference

### Server Management

| Command | What to Say |
|---------|-------------|
| Start server | "Start vLLM with [model-name]" |
| Stop server | "Stop the vLLM server" |
| Restart server | "Restart vLLM" |
| Check status | "Is vLLM running?" |
| View logs | "Show vLLM logs" |
| Platform info | "Check my platform for vLLM" |

### Using the Model

| Command | What to Say |
|---------|-------------|
| Chat | "Ask vLLM: [your question]" |
| Complete text | "Use vLLM to complete: [your prompt]" |
| List models | "What models are available?" |

---

## Platform & Runtime Support

The server automatically detects Podman or Docker (prefers Podman) and selects the appropriate container image:

| Your Platform | Container Runtime | What Happens |
|---------------|-------------------|--------------|
| Linux + NVIDIA GPU | Podman/Docker | Uses GPU-accelerated image |
| Linux (no GPU) | Podman/Docker | Uses CPU image |
| macOS (any) | Podman/Docker | Uses macOS-optimized image |
| Windows + NVIDIA | Docker | Uses GPU via WSL2 |
| Windows (no GPU) | Podman/Docker | Uses CPU image |

**To force a specific runtime**, set the environment variable:
```bash
export VLLM_CONTAINER_RUNTIME=podman  # or "docker"
```

---

## Troubleshooting

### "No container runtime found" or "Podman/Docker is not running"

Install and start Podman or Docker:
- **Podman:** https://podman.io/getting-started/installation
- **Docker:** https://docs.docker.com/engine/install/

### "Model taking too long to load"

Use `get_vllm_logs` to check progress. First-time model downloads can take several minutes.

### "Connection refused"

The model is still loading. Wait and check status again.

### "Out of memory"

Try a smaller model like `TinyLlama/TinyLlama-1.1B-Chat-v1.0` or reduce `max_model_len`.

---

## Next Steps

- See [README.md](README.md) for full documentation
- Explore available [prompts](README.md#prompts) for common tasks
- Learn about [configuration options](README.md#configuration)
