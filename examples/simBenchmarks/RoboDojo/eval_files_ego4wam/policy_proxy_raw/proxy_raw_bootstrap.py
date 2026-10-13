#!/usr/bin/env python3
# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: MIT
"""Register the isolated historical Ego4WAM adapter in a pinned XPolicyLab."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def register_policy() -> None:
    root = Path(os.environ["ROBODOJO_ROOT"]).expanduser().resolve(strict=True)
    for path in (root, root / "XPolicyLab"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import XPolicyLab.policy as policy_package

    proxy_root = str(Path(__file__).resolve().parent / "policy_modules")
    if proxy_root not in policy_package.__path__:
        policy_package.__path__.insert(0, proxy_root)


def main() -> None:
    register_policy()
    from XPolicyLab.setup_policy_server import main as start_policy_server
    from XPolicyLab.setup_policy_server import parse_args_and_config

    config = parse_args_and_config()
    if config.get("policy_name") != "Ego4WAM":
        raise ValueError("This entrypoint serves only policy_name=Ego4WAM")
    start_policy_server(config)


if __name__ == "__main__":
    main()
