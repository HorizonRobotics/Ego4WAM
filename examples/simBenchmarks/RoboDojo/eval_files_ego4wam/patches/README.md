<!--
Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
Author: Ego4WAM Team.
SPDX-License-Identifier: Apache-2.0
-->

# Historical runtime patches

These patches are extracted from the pre-paper local evaluation runtime,
against the revisions checked by `../prepare_runtime.py`. They preserve its
camera capture and rollout behavior rather than backporting the newer
synchronized renderer or adding a policy-side RGB delay.

- `robodojo.patch`: raw camera fields, camera cleanup, original video/render
  path, source asset-path support, capability inventory and restart/resume.
- `xpolicylab.patch`: historical batched RPC, camera-pose RPC and keepalive
  settings. The independent Ego4WAM policy is shipped under `policy_proxy_raw/`;
  this patch does not modify XPolicyLab's starVLA policy.
- `isaaclab.patch`: historical URDF importer version pin in the app definition.

Apply only to the exact pinned checkouts. The preparer refuses a revision or
patch-context mismatch and never resets a user's repository.

Upstream attribution and licenses still apply. The pinned RoboDojo LICENSE is
included as `RoboDojo-LICENSE`; XPolicyLab's Apache license is included as
`XPolicyLab-LICENSE`. The IsaacLab fork retains its own license in its checkout.
These patches do not grant any license to benchmark assets or NVIDIA software.
