"""Shared evaluation entrypoint; model-specific workers use their own envs.

python -m embodiq.entry.evaluate --model openvla --checkpoint /path/to/ckpt ...
"""
from embodiq.entry.run_adapter import main

if __name__ == "__main__":
    raise SystemExit(main())
