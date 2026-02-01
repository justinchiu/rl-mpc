CleanRL Reference Script
========================

This folder vendors the CleanRL single-file DQN script for comparison.

Source
------
- Project: CleanRL
- Version: v1.0.0
- Files: cleanrl/dqn.py, cleanrl/ppo.py
  (vendored as `third_party/cleanrl/v1.0.0/dqn.py` and `third_party/cleanrl/v1.0.0/ppo.py`)
- License: MIT (see `third_party/cleanrl/v1.0.0/LICENSE`)

Notes
-----
- The script is copied verbatim and intentionally unmodified.
- It uses argparse (`--flag` style), not chz `key=value` args.
- It imports `gym`, `stable_baselines3`, and `tensorboard`.

Dependencies
------------
Install the extra dependencies:
```bash
uv pip install gym stable-baselines3 tensorboard
```

Integration test
----------------
Run short sanity checks:
```bash
./scripts/test_cleanrl.sh
```
