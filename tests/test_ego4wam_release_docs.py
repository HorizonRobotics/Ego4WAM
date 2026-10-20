# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: MIT
"""Release documentation links, command syntax and license-scope guards."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_root_license_and_package_metadata_are_mit():
    license_text = (ROOT / "LICENSE").read_text()
    assert license_text.startswith("MIT License\n")
    assert "This repository is a derivative fork of StarVLA." in license_text
    assert "Copyright (c) 2026 Beijing Horizon Information Technology Company Limited." in license_text
    assert "StarVLA-derived code remains unmodified" not in license_text
    assert "two latest upstream StarVLA commits as separate" in license_text
    assert "chore: clone from StarVLA" in license_text
    assert "MIT License" in (ROOT / "starVLA/training/trainer_utils/assets/MIT.txt").read_text()
    metadata = (ROOT / "pyproject.toml").read_text()
    assert 'license = "MIT"' in metadata
    assert "license: MIT\n" in (ROOT / "CITATION.cff").read_text()
    assert '"setuptools>=77"' in metadata
    assert '"License :: OSI Approved :: MIT License"' not in metadata
    for item in ('"NOTICE"', '"LICENSES/*.txt"'):
        assert item in metadata
    notice = (ROOT / "NOTICE").read_text()
    assert "Copyright (c) 2026 Beijing Horizon Information Technology Company Limited." in notice
    assert "Author: Ego4WAM Team." in notice


def test_upstream_license_copies_and_headers_are_retained():
    original = (ROOT / "LICENSES/StarVLA-MIT.txt").read_text()
    assert "MIT License" in original
    assert "Copyright (c) StarVLA Team." in original
    assert "Preserve attribution" in original
    assert "Copyright (c) 2026 The FastWAM Authors." in (ROOT / "LICENSES/FastWAM-MIT.txt").read_text()
    patches = ROOT / "examples/simBenchmarks/RoboDojo/eval_files_ego4wam/patches"
    for name, source in (("RoboDojo-MIT.txt", "RoboDojo-LICENSE"), ("XPolicyLab-Apache-2.0.txt", "XPolicyLab-LICENSE")):
        assert (ROOT / "LICENSES" / name).read_bytes() == (patches / source).read_bytes()
    assert "MIT License" in (ROOT / "LICENSES/DiffusionPolicy-MIT.txt").read_text()
    for file in ("starVLA/model/framework/WM4A/Ego4WAM.py", "starVLA/model/modules/world_model/WanMoT.py"):
        header = (ROOT / file).read_text().splitlines()[:8]
        assert "The FastWAM Authors." in header[0]
        assert "# SPDX-License-Identifier: MIT" in header


def test_ego4wam_authored_headers_include_mit():
    checked = []
    for tree in ("starVLA", "deployment", "examples/simBenchmarks/RoboDojo", "tests"):
        for path in (ROOT / tree).rglob("*"):
            if path.suffix not in {".py", ".sh", ".yaml"}:
                continue
            header = "\n".join(path.read_text().splitlines()[:12])
            if "Copyright (c) 2026 Beijing Horizon Information Technology Company Limited." not in header:
                continue
            assert "Author: Ego4WAM Team." in header, path
            identifier = next(
                line.split("SPDX-License-Identifier:", 1)[1].strip()
                for line in header.splitlines()
                if "SPDX-License-Identifier:" in line
            )
            assert "MIT" in identifier.split(" AND "), path
            checked.append(path)
    assert len(checked) >= 50


@pytest.mark.parametrize(
    "filename",
    [
        "README.md",
        "docs/RoboDojo.md",
        "LICENSES/README.md",
        "examples/simBenchmarks/RoboDojo/eval_files_ego4wam/README.md",
    ],
)
def test_local_document_links_resolve(filename):
    path = ROOT / filename
    for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", path.read_text()):
        if target.startswith(("http://", "https://", "#")):
            continue
        relative = target.split("#", 1)[0]
        assert (path.parent / relative).exists(), (filename, target)


@pytest.mark.parametrize("filename", ["README.md", "docs/RoboDojo.md"])
def test_documented_bash_blocks_parse_without_executing(filename):
    blocks = re.findall(r"```bash\n(.*?)```", (ROOT / filename).read_text(), re.DOTALL)
    assert blocks
    for block in blocks:
        result = subprocess.run(["bash", "-n"], input=block, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_readme_points_to_ego4wam_recipes_and_historical_evaluation():
    text = (ROOT / "README.md").read_text()
    assert "## 🔥 RoboDojo Post-training" in text
    assert "## 🤖 RoboDojo Evaluation" in text
    assert "🏋" not in text
    assert "RoboDojo Posttraining" not in text
    assert "historical RoboDojo version used in the paper" in text
    for filename in ("README.md", "docs/RoboDojo.md", "examples/simBenchmarks/RoboDojo/eval_files_ego4wam/README.md"):
        headings = [line for line in (ROOT / filename).read_text().splitlines() if line.startswith("#")]
        assert all("historical" not in line.lower() for line in headings)
    assert "License-MIT" in text
    assert "License-Apache_2.0" not in text
    for source in ("wan", "video_pretrained"):
        for mode in ("base", "joint"):
            name = f"train_ego4wam_{source}_robodojo_{mode}.sh"
            assert name in text
            assert (ROOT / "examples/simBenchmarks/RoboDojo/train_files_ego4wam" / name).is_file()
    assert "eval_files_ego4wam/local" not in text or "local_server.sh" in text
    for name in ("local_server.sh", "local_client.sh", "summarize_results.py", "NUM_EPISODES=10", "NUM_EPISODES=native"):
        assert name in text
    assert "not the current official leaderboard/submission protocol" in text
    assert "not yet published on HF" not in text
    assert "prepared locally" not in text
    assert "42ea672a744429a98221a321dab3b1fc55b871f2" in text
    assert "--include 'Ego4WAM_VideoPretrained/**'" in text
    assert "--include 'Ego4WAM_VideoPretrained_RoboDojoJoint/**'" in text
    assert 'export CKPT="$EGO4WAM_MODEL_ROOT/Ego4WAM_VideoPretrained_RoboDojoJoint"' in text
    assert 'export EGO4WAM_PRETRAINED_DIR="$EGO4WAM_MODEL_ROOT/Ego4WAM_VideoPretrained"' in text
    assert "RoboDojo/train_files/" not in text
