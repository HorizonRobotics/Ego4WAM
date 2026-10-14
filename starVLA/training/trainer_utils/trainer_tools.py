# Copyright (c) StarVLA Team.
# Modifications Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Ego4WAM additions/modifications: Apache-2.0; upstream StarVLA portions: MIT.
# See LICENSES/StarVLA-MIT.txt for the retained upstream license.

"""Shared command-line normalization used by the Ego4WAM training entrypoint."""


def normalize_dotlist_args(args):
    """
    Convert ['--x.y', 'val'] and ['--flag'] → ['x.y=val', 'flag=true']
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        else:
            pass  # skip orphaned values
    return normalized
