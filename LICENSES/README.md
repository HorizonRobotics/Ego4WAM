<!--
Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
Author: Ego4WAM Team.
SPDX-License-Identifier: MIT
-->

# License scope and upstream notices

Ego4WAM-authored additions and modifications use [MIT](../LICENSE).
Their rights holder is **Beijing Horizon Information Technology Company Limited**;
author attribution remains **Ego4WAM Team**.
The inherited code and third-party components retain their original licenses.
This is not a blanket relicensing of upstream code.

| Component | Retained terms | License text |
| --- | --- | --- |
| StarVLA infrastructure | Original StarVLA MIT text, including its attribution instructions | [StarVLA-MIT.txt](StarVLA-MIT.txt) |
| FastWAM-derived experts, framework, MoT and scheduler | MIT; retain the upstream copyright alongside Ego4WAM modification notices | [FastWAM-MIT.txt](FastWAM-MIT.txt) |
| RoboDojo evaluation patches | MIT | [RoboDojo-MIT.txt](RoboDojo-MIT.txt) |
| XPolicyLab-derived adapter portions | Apache-2.0 for inherited portions; MIT for Ego4WAM modifications | [XPolicyLab-Apache-2.0.txt](XPolicyLab-Apache-2.0.txt) |
| NVIDIA/GR00T-derived dataset readers and transforms | Apache-2.0 with their original source notices | [Apache-2.0.txt](Apache-2.0.txt) and the source headers |
| Diffusion Policy provenance in retained dataset utilities; unused vendored model code removed | MIT | [DiffusionPolicy-MIT.txt](DiffusionPolicy-MIT.txt) |
| OpenVLA/Prismatic reference for the inherited overwatch logger | Supplemental MIT license archive; inherited source headers remain unchanged | [OpenVLA-MIT.txt](OpenVLA-MIT.txt) |
| OpenPI client reference for inherited msgpack_numpy | Supplemental Apache-2.0 license archive; no inferred file-level notice is added | [OpenPI-Apache-2.0.txt](OpenPI-Apache-2.0.txt) |
| Original msgpack-numpy reference | Supplemental BSD-3-Clause license archive | [MsgpackNumpy-BSD-3-Clause.txt](MsgpackNumpy-BSD-3-Clause.txt) |

Headers inherited from StarVLA are preserved as received. Original copyright,
license and modification statements are retained verbatim; files without
such a header do not receive an inferred copyright, SPDX identifier, or
source-provenance notice. Ego4WAM functionality changes in those files are
not reverted by this header policy. Supplemental license archives remain
available separately without changing the inherited source-file headers.
The StarVLA copyright quoted in NOTICE comes from its original repository
LICENSE, not an inference about all individual files.

New Ego4WAM-authored files and newly introduced MIT-derived files use
`SPDX-License-Identifier: MIT`. Newly introduced adapter files that also retain
Apache-licensed third-party portions use `MIT AND Apache-2.0`. These mixed
expressions retain the applicable terms for each portion; they are not a
choice of license and do not turn third-party components into MIT originals.
Inherited StarVLA headers are not rewritten to impose these conventions.
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

New training exports use a scoped MIT notice for Ego4WAM contributions and
retain original backbone terms under `model_assets/licenses/`. The Wan2.2 and
Google UMT5 models declare Apache-2.0 in their original model cards. Existing
published HF bundles retain their published licenses; this repository change
does not rewrite or relicense them.
