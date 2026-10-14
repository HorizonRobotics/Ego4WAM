<!--
Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
Author: Ego4WAM Team.
SPDX-License-Identifier: Apache-2.0
-->

# License scope and upstream notices

Ego4WAM-authored additions and modifications use [Apache-2.0](../LICENSE).
Their rights holder is **Beijing Horizon Information Technology Company Limited**;
author attribution remains **Ego4WAM Team**.
The inherited code and third-party components retain their original licenses.
This is not a blanket relicensing of upstream code.

| Component | Retained terms | License text |
| --- | --- | --- |
| StarVLA infrastructure | Original StarVLA MIT text, including its attribution instructions | [StarVLA-MIT.txt](StarVLA-MIT.txt) |
| FastWAM-derived experts, framework, MoT and scheduler | MIT for the inherited portions; Apache-2.0 for Ego4WAM modifications | [FastWAM-MIT.txt](FastWAM-MIT.txt) |
| RoboDojo evaluation patches | MIT | [RoboDojo-MIT.txt](RoboDojo-MIT.txt) |
| XPolicyLab components and integration | Apache-2.0 | [XPolicyLab-Apache-2.0.txt](XPolicyLab-Apache-2.0.txt) |
| NVIDIA/GR00T-derived dataset readers and transforms | Apache-2.0 with their original source notices | [Apache-2.0](../LICENSE) and the source headers |
| Diffusion Policy provenance in retained dataset utilities; unused vendored model code removed | MIT | [DiffusionPolicy-MIT.txt](DiffusionPolicy-MIT.txt) |
| OpenVLA/Prismatic-derived overwatch logger | MIT; retained original source attribution | [OpenVLA-MIT.txt](OpenVLA-MIT.txt) |
| OpenPI-client-derived msgpack_numpy implementation | Apache-2.0; retains msgpack-numpy adaptation provenance | [OpenPI-Apache-2.0.txt](OpenPI-Apache-2.0.txt) |
| Original msgpack-numpy provenance | BSD-3-Clause and its original copyright notice | [MsgpackNumpy-BSD-3-Clause.txt](MsgpackNumpy-BSD-3-Clause.txt) |

File-level copyright is retained only where supported by the original source
notice or a verified original license. Obtaining a file from StarVLA does not
establish StarVLA as its copyright holder. Files without an original owner
notice use `Source provenance: StarVLA project ...` instead; the preserved
repository license is referenced separately. The StarVLA copyright quoted in
NOTICE comes from its original repository LICENSE, not an inference about all
individual files.

Mixed-origin files use `SPDX-License-Identifier: Apache-2.0 AND MIT` where
Ego4WAM's Apache-2.0 additions and upstream MIT portions coexist. This means
the applicable notices for both portions are retained, not a choice of license.
Unmodified upstream headers and the licenses beside evaluation patches remain
unchanged. See [NOTICE](../NOTICE) for attribution.

Ego4WAM-authored additions/modifications identify the company copyright and
the Ego4WAM Team author attribution. Unmodified upstream implementations do
not gain a company copyright merely through notice maintenance. JSON, patches, task lists
and the robot description use adjacent `<filename>.license` notices so their
machine-readable contents remain unchanged. These notices cover only our
additions/modifications and packaging; they do not relicense benchmark data
or third-party assets. Standard license texts and retained upstream license
copies are deliberately kept verbatim, without new ownership headers.

The pruned ARX X5A URDF is derived from a third-party robot asset. Its exact
source license still requires provenance confirmation; the company notice
does not claim ownership of the original robot description or grant rights
to redistribute it.

This inventory describes the license notices retained by this code release;
external dependencies, models, datasets, simulator SDKs and assets keep their
own terms. Do not treat the repository's license as permission to redistribute
those external resources.
