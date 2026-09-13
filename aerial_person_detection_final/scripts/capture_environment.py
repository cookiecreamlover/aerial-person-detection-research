"""Capture the environment metadata required for latency claims."""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    try:
        import torch
        torch_info = {'version': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda}
        if torch.cuda.is_available():
            torch_info['device_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        torch_info = {'not_installed': True}
    try:
        import ultralytics
        ultralytics_version = getattr(ultralytics, '__version__', None)
    except ImportError:
        ultralytics_version = None
    result = {
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': __import__('os').cpu_count(),
        'memory_summary': command_output(['bash', '-lc', "free -h | sed -n '2p'"]),
        'gpu_summary': command_output(['bash', '-lc', 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null']),
        'pytorch': torch_info,
        'ultralytics_version': ultralytics_version,
        'torchvision_version': _torchvision_version(),
    }
    output = Path('results/environment.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f'Wrote {output}')


def _torchvision_version():
    try:
        import torchvision
        return getattr(torchvision, '__version__', None)
    except ImportError:
        return None


if __name__ == '__main__':
    main()
