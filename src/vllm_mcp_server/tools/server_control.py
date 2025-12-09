"""Server control tools for vLLM MCP Server (container mode).

Supports both Podman and Docker with platform-specific configurations:
- Linux: Full NVIDIA GPU support
- macOS (Apple Silicon): CPU-only mode (no NVIDIA GPU in containers)
- macOS (Intel): CPU-only mode
- Windows: GPU support via WSL2 + NVIDIA Container Toolkit
"""

import asyncio
import platform
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from mcp.types import TextContent

from vllm_mcp_server.utils.config import get_settings


class Platform(Enum):
    """Supported platforms."""
    LINUX = "linux"
    MACOS_ARM = "macos_arm"
    MACOS_INTEL = "macos_intel"
    WINDOWS = "windows"
    UNKNOWN = "unknown"


class ContainerRuntime(Enum):
    """Supported container runtimes."""
    PODMAN = "podman"
    DOCKER = "docker"
    NONE = "none"


@dataclass
class PlatformInfo:
    """Platform-specific information."""
    platform: Platform
    container_runtime: ContainerRuntime
    has_nvidia_gpu: bool
    runtime_available: bool
    runtime_running: bool
    cache_path: str
    gpu_flags: list[str]
    notes: list[str]


def _detect_platform() -> Platform:
    """Detect the current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        return Platform.LINUX
    elif system == "darwin":
        if machine in ("arm64", "aarch64"):
            return Platform.MACOS_ARM
        return Platform.MACOS_INTEL
    elif system == "windows":
        return Platform.WINDOWS
    return Platform.UNKNOWN


async def _run_command(cmd: list[str], timeout: float = 30.0) -> tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
        return (
            process.returncode or 0,
            stdout.decode("utf-8"),
            stderr.decode("utf-8"),
        )
    except asyncio.TimeoutError:
        return (1, "", "Command timed out")
    except Exception as e:
        return (1, "", str(e))


async def _detect_container_runtime() -> tuple[ContainerRuntime, bool, bool, str]:
    """
    Detect available container runtime (Podman preferred over Docker).
    
    Returns:
        Tuple of (runtime, is_available, is_running, error_message)
    """
    settings = get_settings()
    
    # Check user preference
    preferred = settings.container_runtime.lower() if settings.container_runtime else "auto"
    
    # Try Podman first (unless Docker is explicitly preferred)
    if preferred in ("auto", "podman"):
        if shutil.which("podman"):
            exit_code, _, stderr = await _run_command(["podman", "info"])
            if exit_code == 0:
                return ContainerRuntime.PODMAN, True, True, ""
            elif preferred == "podman":
                return ContainerRuntime.PODMAN, True, False, f"Podman is not running: {stderr.strip()}"
    
    # Try Docker
    if preferred in ("auto", "docker"):
        if shutil.which("docker"):
            exit_code, _, stderr = await _run_command(["docker", "info"])
            if exit_code == 0:
                return ContainerRuntime.DOCKER, True, True, ""
            elif preferred == "docker":
                return ContainerRuntime.DOCKER, True, False, f"Docker is not running: {stderr.strip()}"
    
    # Nothing available
    if preferred == "podman":
        return ContainerRuntime.NONE, False, False, "Podman is not installed or not in PATH"
    elif preferred == "docker":
        return ContainerRuntime.NONE, False, False, "Docker is not installed or not in PATH"
    else:
        return ContainerRuntime.NONE, False, False, "No container runtime (Podman or Docker) found"


async def _check_nvidia_gpu(runtime: ContainerRuntime) -> bool:
    """Check if NVIDIA GPU is available."""
    # Try nvidia-smi first
    if shutil.which("nvidia-smi"):
        exit_code, _, _ = await _run_command(["nvidia-smi", "-L"])
        if exit_code == 0:
            return True

    # Check container runtime for NVIDIA support
    if runtime == ContainerRuntime.DOCKER:
        exit_code, stdout, _ = await _run_command(
            ["docker", "info", "--format", "{{.Runtimes}}"]
        )
        if exit_code == 0 and "nvidia" in stdout.lower():
            return True
    elif runtime == ContainerRuntime.PODMAN:
        # Podman uses CDI (Container Device Interface) for GPU access
        # Check if nvidia-container-toolkit is available
        exit_code, stdout, _ = await _run_command(
            ["podman", "info", "--format", "{{.Host.Security.SECCOMPEnabled}}"]
        )
        # If podman works and nvidia-smi exists, assume GPU support
        if exit_code == 0 and shutil.which("nvidia-smi"):
            return True

    return False


def _get_cache_path(plat: Platform) -> str:
    """Get the HuggingFace cache path for the platform."""
    home = Path.home()
    
    if plat == Platform.WINDOWS:
        # Windows path format for containers
        cache_dir = home / ".cache" / "huggingface"
        # Convert to container-compatible path (e.g., //c/Users/...)
        return str(cache_dir).replace("\\", "/")
    else:
        return str(home / ".cache" / "huggingface")


def _get_runtime_cmd(runtime: ContainerRuntime) -> str:
    """Get the command for the container runtime."""
    if runtime == ContainerRuntime.PODMAN:
        return "podman"
    return "docker"


async def get_platform_info() -> PlatformInfo:
    """Get comprehensive platform information."""
    plat = _detect_platform()
    runtime, runtime_available, runtime_running, _ = await _detect_container_runtime()
    has_nvidia = await _check_nvidia_gpu(runtime) if runtime_running else False
    
    notes: list[str] = []
    gpu_flags: list[str] = []
    
    # Runtime info
    if runtime == ContainerRuntime.PODMAN:
        notes.append("Using Podman as container runtime")
    elif runtime == ContainerRuntime.DOCKER:
        notes.append("Using Docker as container runtime")
    else:
        notes.append("No container runtime available")
    
    if plat == Platform.LINUX:
        if has_nvidia:
            if runtime == ContainerRuntime.PODMAN:
                # Podman uses --device for GPU access with CDI
                gpu_flags = ["--device", "nvidia.com/gpu=all"]
            else:
                gpu_flags = ["--gpus", "all"]
            notes.append("NVIDIA GPU detected - full GPU acceleration available")
        else:
            notes.append("No NVIDIA GPU detected - running in CPU mode")
            
    elif plat == Platform.MACOS_ARM:
        notes.append("Apple Silicon detected - containers run in CPU mode")
        notes.append("For GPU acceleration, consider running vLLM natively with Metal")
        
    elif plat == Platform.MACOS_INTEL:
        notes.append("Intel Mac detected - containers run in CPU mode")
        
    elif plat == Platform.WINDOWS:
        if has_nvidia:
            gpu_flags = ["--gpus", "all"]
            notes.append("NVIDIA GPU detected via WSL2 - GPU acceleration available")
        else:
            notes.append("No NVIDIA GPU detected - running in CPU mode")
            notes.append("Ensure WSL2 and NVIDIA Container Toolkit are installed for GPU support")
    
    return PlatformInfo(
        platform=plat,
        container_runtime=runtime,
        has_nvidia_gpu=has_nvidia,
        runtime_available=runtime_available,
        runtime_running=runtime_running,
        cache_path=_get_cache_path(plat),
        gpu_flags=gpu_flags,
        notes=notes,
    )


async def _is_container_running(container_name: str, runtime: ContainerRuntime) -> bool:
    """Check if a container is running."""
    cmd = _get_runtime_cmd(runtime)
    exit_code, stdout, _ = await _run_command([
        cmd, "ps", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"
    ])
    return exit_code == 0 and container_name in stdout.strip().split("\n")


async def _is_container_exists(container_name: str, runtime: ContainerRuntime) -> bool:
    """Check if a container exists (running or stopped)."""
    cmd = _get_runtime_cmd(runtime)
    exit_code, stdout, _ = await _run_command([
        cmd, "ps", "-a", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"
    ])
    return exit_code == 0 and container_name in stdout.strip().split("\n")


async def start_vllm(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Start a vLLM server in a container with platform-aware configuration.

    Args:
        arguments: Dictionary containing:
            - model: Model to serve (required)
            - port: Port to expose (default: 8000)
            - gpu_memory_utilization: GPU memory fraction (default: 0.9)
            - cpu_only: Force CPU mode even if GPU available (default: False)
            - tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
            - max_model_len: Maximum model context length (optional)
            - dtype: Data type (auto, float16, bfloat16, float32)
            - container_name: Name for the container
            - extra_args: Additional vLLM arguments

    Returns:
        List of TextContent with the result.
    """
    settings = get_settings()
    
    # Get platform info
    platform_info = await get_platform_info()
    runtime_cmd = _get_runtime_cmd(platform_info.container_runtime)
    
    if not platform_info.runtime_available:
        return [TextContent(
            type="text",
            text="❌ Error: No container runtime found.\n\n"
                 "Please install Podman or Docker:\n"
                 "- Podman: https://podman.io/getting-started/installation\n"
                 "- Docker: https://docs.docker.com/engine/install/"
        )]
    
    if not platform_info.runtime_running:
        runtime_name = platform_info.container_runtime.value.capitalize()
        return [TextContent(
            type="text",
            text=f"❌ Error: {runtime_name} is not running.\n\n"
                 f"Please start {runtime_name}."
        )]

    model = arguments.get("model")
    if not model:
        return [TextContent(type="text", text="❌ Error: 'model' is required to start vLLM server")]

    port = arguments.get("port", 8000)
    gpu_memory = arguments.get("gpu_memory_utilization", settings.gpu_memory_utilization)
    cpu_only = arguments.get("cpu_only", False)
    tensor_parallel_size = arguments.get("tensor_parallel_size", 1)
    max_model_len = arguments.get("max_model_len")
    dtype = arguments.get("dtype", "auto")
    extra_args = arguments.get("extra_args", [])
    container_name = arguments.get("container_name", settings.container_name)

    # Check if already running
    if await _is_container_running(container_name, platform_info.container_runtime):
        return [TextContent(
            type="text",
            text=f"⚠️ Container '{container_name}' is already running.\n\n"
                 f"Options:\n"
                 f"- Stop it first with `stop_vllm`\n"
                 f"- Use a different `container_name`\n"
                 f"- Use `restart_vllm` to restart it"
        )]
    
    # Remove existing stopped container with same name
    if await _is_container_exists(container_name, platform_info.container_runtime):
        await _run_command([runtime_cmd, "rm", container_name])

    # Determine GPU mode
    use_gpu = platform_info.has_nvidia_gpu and not cpu_only and platform_info.gpu_flags
    
    # Set default max_model_len based on GPU/CPU mode if not specified
    if max_model_len is None:
        if use_gpu:
            max_model_len = 8096  # Default for GPU mode
        else:
            max_model_len = 2048  # Default for CPU mode (matches max_num_batched_tokens)
    
    # Select appropriate container image based on platform
    docker_image = arguments.get("docker_image")  # Allow override
    if not docker_image:
        if platform_info.platform in (Platform.MACOS_ARM, Platform.MACOS_INTEL):
            # macOS uses the macOS-specific image
            docker_image = settings.docker_image_macos
        elif use_gpu:
            # GPU mode uses the default vLLM image
            docker_image = settings.docker_image
        else:
            # CPU mode (Linux x86_64, Windows without GPU)
            docker_image = settings.docker_image_cpu

    # Check if using macOS/CPU image that requires environment variable configuration
    use_env_config = (
        platform_info.platform in (Platform.MACOS_ARM, Platform.MACOS_INTEL) 
        and not arguments.get("docker_image")
    ) or docker_image in (settings.docker_image_macos, settings.docker_image_cpu)
    
    # Determine container home directory based on image type
    # macOS/CPU images run as user 'vllm' (not root), so use /home/vllm
    container_hf_home = "/home/vllm/.cache/huggingface" if use_env_config else "/root/.cache/huggingface"
    
    # Build container run command
    cmd = [
        runtime_cmd, "run",
        "-d",  # Detached mode
        "--name", container_name,
        "-p", f"{port}:8000",
        "--shm-size", "8g",  # Shared memory for PyTorch
        "-e", f"HF_HOME={container_hf_home}",
    ]
    
    # Add HuggingFace token if configured (required for gated models)
    if settings.hf_token:
        cmd.extend(["-e", f"HF_TOKEN={settings.hf_token}"])
    
    if use_env_config:
        # Configure via environment variables for macOS/CPU images
        # These images use a startup script that reads VLLM_* env vars
        cmd.extend(["-e", f"VLLM_MODEL={model}"])
        
        # Use bfloat16 for CPU mode (supported by the image)
        if dtype == "auto":
            dtype = "bfloat16"
        cmd.extend(["-e", f"VLLM_DTYPE={dtype}"])
        
        if max_model_len:
            cmd.extend(["-e", f"VLLM_MAX_MODEL_LEN={max_model_len}"])
        
        # CPU-specific settings
        cpu_kvcache_space = arguments.get("cpu_kvcache_space", 4)
        cmd.extend(["-e", f"VLLM_CPU_KVCACHE_SPACE={cpu_kvcache_space}"])
    else:
        # GPU mode - will use command-line arguments after the image
        pass
    
    # Add GPU flags if available
    if use_gpu:
        cmd.extend(platform_info.gpu_flags)
    
    # Add volume mount for HuggingFace cache
    cmd.extend(["-v", f"{platform_info.cache_path}:{container_hf_home}"])
    
    # Add the container image
    cmd.append(docker_image)
    
    # For GPU images, add vLLM command-line arguments after the image
    if not use_env_config:
        cmd.extend(["--model", model])
        
        if use_gpu:
            cmd.extend(["--gpu-memory-utilization", str(gpu_memory)])
            if tensor_parallel_size > 1:
                cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
        else:
            cmd.extend(["--device", "cpu"])
            if dtype == "auto":
                dtype = "float32"
        
        if max_model_len:
            cmd.extend(["--max-model-len", str(max_model_len)])
        
        cmd.extend(["--dtype", dtype])
        
        # Add extra arguments
        if extra_args:
            if isinstance(extra_args, list):
                cmd.extend(extra_args)
            else:
                cmd.extend(str(extra_args).split())

    # Run the container
    exit_code, stdout, stderr = await _run_command(cmd, timeout=60.0)

    if exit_code != 0:
        return [TextContent(
            type="text",
            text=f"❌ Failed to start vLLM container:\n\n```\n{stderr}\n```\n\n"
                 f"**Command:** `{' '.join(cmd)}`"
        )]

    container_id = stdout.strip()[:12]
    
    # Build result message
    mode = "GPU" if use_gpu else "CPU"
    gpu_info = ""
    if use_gpu:
        gpu_info = f"- GPU Memory: {gpu_memory * 100:.0f}%\n"
        if tensor_parallel_size > 1:
            gpu_info += f"- Tensor Parallel: {tensor_parallel_size} GPUs\n"
    
    platform_notes = "\n".join(f"  - {note}" for note in platform_info.notes)
    runtime_name = platform_info.container_runtime.value.capitalize()
    
    return [TextContent(
        type="text",
        text=f"✅ vLLM server started successfully!\n\n"
             f"**Container Info:**\n"
             f"- Runtime: **{runtime_name}**\n"
             f"- Container ID: `{container_id}`\n"
             f"- Container Name: `{container_name}`\n"
             f"- Image: `{docker_image}`\n"
             f"- Model: `{model}`\n"
             f"- Mode: **{mode}**\n"
             f"- Port: {port}\n"
             f"- Data Type: {dtype}\n"
             f"{gpu_info}"
             f"\n**API Endpoints:**\n"
             f"- Base URL: http://localhost:{port}\n"
             f"- OpenAI API: http://localhost:{port}/v1\n"
             f"- Health: http://localhost:{port}/health\n"
             f"\n**Platform ({platform_info.platform.value}):**\n{platform_notes}\n"
             f"\n⏳ **Note:** The model may take several minutes to download and load.\n"
             f"Use `vllm_status` to check when it's ready, or `get_vllm_logs` to see progress."
    )]


async def stop_vllm(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Stop a running vLLM container.

    Args:
        arguments: Dictionary containing:
            - container_name: Name of container to stop (default: from settings)
            - remove: Whether to remove the container (default: True)
            - timeout: Seconds to wait before killing (default: 10)

    Returns:
        List of TextContent with the result.
    """
    settings = get_settings()
    
    platform_info = await get_platform_info()
    if not platform_info.runtime_running:
        runtime_name = platform_info.container_runtime.value.capitalize() if platform_info.container_runtime != ContainerRuntime.NONE else "Container runtime"
        return [TextContent(type="text", text=f"❌ Error: {runtime_name} is not running.")]

    runtime_cmd = _get_runtime_cmd(platform_info.container_runtime)
    container_name = arguments.get("container_name", settings.container_name)
    remove = arguments.get("remove", True)
    timeout = arguments.get("timeout", 10)

    # Check if running
    is_running = await _is_container_running(container_name, platform_info.container_runtime)
    exists = await _is_container_exists(container_name, platform_info.container_runtime)
    
    if not exists:
        return [TextContent(
            type="text",
            text=f"ℹ️ Container '{container_name}' does not exist."
        )]
    
    result_parts = []
    
    if is_running:
        # Stop container with timeout
        exit_code, _, stderr = await _run_command(
            [runtime_cmd, "stop", "-t", str(timeout), container_name]
        )
        if exit_code != 0:
            return [TextContent(
                type="text",
                text=f"❌ Failed to stop container: {stderr}"
            )]
        result_parts.append(f"✅ Container '{container_name}' stopped.")
    else:
        result_parts.append(f"ℹ️ Container '{container_name}' was not running.")

    # Remove container if requested
    if remove:
        exit_code, _, stderr = await _run_command([runtime_cmd, "rm", container_name])
        if exit_code == 0:
            result_parts.append(f"✅ Container '{container_name}' removed.")
        else:
            result_parts.append(f"⚠️ Failed to remove container: {stderr}")

    return [TextContent(type="text", text="\n".join(result_parts))]


async def restart_vllm(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Restart a vLLM container.

    Args:
        arguments: Dictionary containing:
            - container_name: Name of container to restart (default: from settings)

    Returns:
        List of TextContent with the result.
    """
    settings = get_settings()
    
    platform_info = await get_platform_info()
    if not platform_info.runtime_running:
        runtime_name = platform_info.container_runtime.value.capitalize() if platform_info.container_runtime != ContainerRuntime.NONE else "Container runtime"
        return [TextContent(type="text", text=f"❌ Error: {runtime_name} is not running.")]

    runtime_cmd = _get_runtime_cmd(platform_info.container_runtime)
    container_name = arguments.get("container_name", settings.container_name)

    if not await _is_container_exists(container_name, platform_info.container_runtime):
        return [TextContent(
            type="text",
            text=f"❌ Container '{container_name}' does not exist.\n"
                 f"Use `start_vllm` to create a new container."
        )]

    exit_code, _, stderr = await _run_command([runtime_cmd, "restart", container_name])
    
    if exit_code != 0:
        return [TextContent(
            type="text",
            text=f"❌ Failed to restart container: {stderr}"
        )]

    return [TextContent(
        type="text",
        text=f"✅ Container '{container_name}' restarted.\n\n"
             f"⏳ The model may take a minute to reload. Use `vllm_status` to check."
    )]


async def list_vllm_containers(arguments: dict[str, Any]) -> list[TextContent]:
    """
    List all vLLM-related containers.

    Args:
        arguments: Dictionary containing:
            - all: Show all containers including stopped (default: False)

    Returns:
        List of TextContent with container information.
    """
    platform_info = await get_platform_info()
    if not platform_info.runtime_running:
        runtime_name = platform_info.container_runtime.value.capitalize() if platform_info.container_runtime != ContainerRuntime.NONE else "Container runtime"
        return [TextContent(type="text", text=f"❌ Error: {runtime_name} is not running.")]

    runtime_cmd = _get_runtime_cmd(platform_info.container_runtime)
    show_all = arguments.get("all", False)
    
    cmd = [runtime_cmd, "ps"]
    if show_all:
        cmd.append("-a")
    cmd.extend([
        "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"
    ])
    
    exit_code, stdout, stderr = await _run_command(cmd)
    
    if exit_code != 0:
        return [TextContent(type="text", text=f"❌ Error listing containers: {stderr}")]
    
    if not stdout.strip() or stdout.strip() == "NAMES\tSTATUS\tPORTS\tIMAGE":
        return [TextContent(
            type="text",
            text="ℹ️ No containers found.\n\nUse `start_vllm` to create one."
        )]
    
    runtime_name = platform_info.container_runtime.value.capitalize()
    return [TextContent(
        type="text",
        text=f"## {runtime_name} Containers\n\n```\n{stdout}\n```"
    )]


async def get_vllm_logs(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Get logs from a vLLM container.

    Args:
        arguments: Dictionary containing:
            - container_name: Name of container (default: from settings)
            - tail: Number of lines to show (default: 50)
            - follow: Whether to show note about following (default: False)

    Returns:
        List of TextContent with container logs.
    """
    settings = get_settings()
    
    platform_info = await get_platform_info()
    if not platform_info.runtime_running:
        runtime_name = platform_info.container_runtime.value.capitalize() if platform_info.container_runtime != ContainerRuntime.NONE else "Container runtime"
        return [TextContent(type="text", text=f"❌ Error: {runtime_name} is not running.")]

    runtime_cmd = _get_runtime_cmd(platform_info.container_runtime)
    container_name = arguments.get("container_name", settings.container_name)
    tail = arguments.get("tail", 50)

    if not await _is_container_exists(container_name, platform_info.container_runtime):
        return [TextContent(
            type="text",
            text=f"❌ Container '{container_name}' does not exist."
        )]

    exit_code, stdout, stderr = await _run_command(
        [runtime_cmd, "logs", "--tail", str(tail), container_name]
    )
    
    if exit_code != 0:
        return [TextContent(type="text", text=f"❌ Error getting logs: {stderr}")]
    
    # Combine stdout and stderr (vLLM logs to stderr)
    logs = stdout + stderr
    
    return [TextContent(
        type="text",
        text=f"## Logs for '{container_name}' (last {tail} lines)\n\n```\n{logs}\n```\n\n"
             f"💡 **Tip:** Run `{runtime_cmd} logs -f {container_name}` in terminal to follow logs in real-time."
    )]


async def get_platform_status(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Get detailed platform and container runtime status information.

    Returns:
        List of TextContent with platform information.
    """
    platform_info = await get_platform_info()
    
    # Platform emoji
    platform_emoji = {
        Platform.LINUX: "🐧",
        Platform.MACOS_ARM: "🍎",
        Platform.MACOS_INTEL: "🍎",
        Platform.WINDOWS: "🪟",
        Platform.UNKNOWN: "❓",
    }
    
    emoji = platform_emoji.get(platform_info.platform, "❓")
    
    # Runtime status
    if platform_info.container_runtime == ContainerRuntime.NONE:
        runtime_status = "❌ Not installed"
        runtime_name = "None"
    elif platform_info.runtime_running:
        runtime_status = "✅ Running"
        runtime_name = platform_info.container_runtime.value.capitalize()
    else:
        runtime_status = "⚠️ Installed but not running"
        runtime_name = platform_info.container_runtime.value.capitalize()
    
    gpu_status = "✅ Available" if platform_info.has_nvidia_gpu else "❌ Not available"
    
    notes_text = "\n".join(f"  - {note}" for note in platform_info.notes) if platform_info.notes else "  - None"
    
    return [TextContent(
        type="text",
        text=f"## Platform Status {emoji}\n\n"
             f"**Platform:** {platform_info.platform.value}\n"
             f"**Container Runtime:** {runtime_name} ({runtime_status})\n"
             f"**NVIDIA GPU:** {gpu_status}\n"
             f"**HF Cache Path:** `{platform_info.cache_path}`\n"
             f"**GPU Flags:** `{' '.join(platform_info.gpu_flags) or 'None (CPU mode)'}`\n"
             f"\n**Notes:**\n{notes_text}"
    )]
