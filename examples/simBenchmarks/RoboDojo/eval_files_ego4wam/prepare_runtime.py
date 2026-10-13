#!/usr/bin/env python3
# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: Apache-2.0
"""Verify pinned source checkouts and apply the historical runtime patches.

Never resets, checks out, or downloads over a user's existing worktree. By
default this is a read-only check; --apply opts into the listed source patches.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REVISIONS = {
    ".": "9b4cc885e8f530ed3ab14a30a312ae70242771c4",
    "XPolicyLab": "fe71eb54675cef495fea817a637386a4f4529153",
    "third_party/IsaacLab": "afca7b09d60d8beb9c1cb28b43066499940b969b",
    "third_party/curobo": "d17b54ce32cba095c0b000c4c58777075d11de0e",
}
PATCHES = {
    ".": "robodojo.patch",
    "XPolicyLab": "xpolicylab.patch",
    "third_party/IsaacLab": "isaaclab.patch",
}


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def prepare(root: Path, *, apply: bool = False) -> None:
    patch_root = Path(__file__).resolve().parent / "patches"
    pending = []
    for directory, expected in REVISIONS.items():
        checkout = root / directory
        revision = git(checkout, "rev-parse", "HEAD")
        if revision.returncode or revision.stdout.strip() != expected:
            raise RuntimeError(f"{checkout}: expected revision {expected}, got {revision.stdout.strip()!r}")
        if directory not in PATCHES:
            continue
        patch = patch_root / PATCHES[directory]
        if git(checkout, "apply", "--reverse", "--check", str(patch)).returncode == 0:
            print(f"Already applied: {patch.name}")
            continue
        check = git(checkout, "apply", "--check", str(patch))
        if check.returncode:
            raise RuntimeError(f"Cannot safely apply {patch.name}: {check.stderr}")
        pending.append((checkout, patch))
    # Validate all revisions and patch contexts before making any change.
    if pending and not apply:
        raise RuntimeError("Historical patches are missing; inspect patches/ and rerun with --apply")
    for checkout, patch in pending:
        result = git(checkout, "apply", str(patch))
        if result.returncode:
            raise RuntimeError(f"Could not apply {patch.name}: {result.stderr}")
        print(f"Applied: {patch.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    prepare(args.robodojo_root.expanduser().resolve(strict=True), apply=args.apply)


if __name__ == "__main__":
    main()
