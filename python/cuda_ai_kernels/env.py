import platform

import torch


def collect_env() -> dict[str, str | int | None]:
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": str(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(device) if device is not None else None,
        "capability": ".".join(map(str, torch.cuda.get_device_capability(device))) if device is not None else None,
    }

