# License scope and upstream notices

Ego4WAM-authored additions and modifications use [Apache-2.0](../LICENSE).
The inherited code and third-party components retain their original licenses.
This is not a blanket relicensing of upstream code.

| Component | Retained terms | License text |
| --- | --- | --- |
| StarVLA infrastructure | Original StarVLA MIT text, including its attribution instructions | [StarVLA-MIT.txt](StarVLA-MIT.txt) |
| FastWAM-derived experts, framework, MoT and scheduler | MIT for the inherited portions; Apache-2.0 for Ego4WAM modifications | [FastWAM-MIT.txt](FastWAM-MIT.txt) |
| RoboDojo evaluation patches | MIT | [RoboDojo-MIT.txt](RoboDojo-MIT.txt) |
| XPolicyLab components and integration | Apache-2.0 | [XPolicyLab-Apache-2.0.txt](XPolicyLab-Apache-2.0.txt) |
| NVIDIA/GR00T-derived dataset readers and transforms | Apache-2.0 with their original source notices | [Apache-2.0](../LICENSE) and the source headers |
| Vendored Diffusion Policy code | MIT | [Original LICENSE](../starVLA/model/framework/VM4A/_dp_vendor/LICENSE) |

Mixed-origin files use `SPDX-License-Identifier: Apache-2.0 AND MIT` where
Ego4WAM's Apache-2.0 additions and upstream MIT portions coexist. This means
the applicable notices for both portions are retained, not a choice of license.
Unmodified upstream headers and the licenses beside evaluation patches remain
unchanged. See [NOTICE](../NOTICE) for attribution.

This inventory describes the license notices retained by this code release;
external dependencies, models, datasets, simulator SDKs and assets keep their
own terms. Do not treat the repository's license as permission to redistribute
those external resources.
