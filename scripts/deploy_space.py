"""Deploy the demo to a Hugging Face Space, pinned to one exact commit of this repo.

    python scripts/deploy_space.py                          # Space Punith25/recon3d, current commit
    python scripts/deploy_space.py --dry-run                # show what would be uploaded

Why pin a commit: the Space installs the recon3d package from GitHub. With "@main" a later push would
silently change the running app on its next restart; with "@<sha>" the Space runs exactly what was
tested and deployed, and redeploying an older sha is a rollback.

The Space must exist already (create it in the browser: SDK Gradio, hardware ZeroGPU), because free
accounts can only pick ZeroGPU there.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/Punithb2/recon3d"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="Punith25/recon3d")
    ap.add_argument("--ref", default="HEAD", help="git commit/tag to deploy (must be pushed to GitHub)")
    ap.add_argument("--model-revision", help="set the Space variable RECON3D_MODEL_REVISION (e.g. v1.0.1)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="skip the uncommitted-changes check")
    args = ap.parse_args()

    sha = git("rev-parse", args.ref)
    if not args.allow_dirty and git("status", "--porcelain"):
        print("You have uncommitted changes. Commit and push first (or pass --allow-dirty).")
        return 1
    remote_branches = git("branch", "-r", "--contains", sha)
    if not remote_branches:
        print(f"Commit {sha[:7]} is not on GitHub yet: run `git push` first (the Space installs it from there).")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "space"
        shutil.copytree(ROOT / "space", stage, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (stage / "requirements.txt").write_text(
            "# Written by scripts/deploy_space.py\n"
            f"recon3d[serve,hub] @ git+{GITHUB}@{sha}\n")
        files = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file())
        print(f"deploying commit {sha[:7]} to https://huggingface.co/spaces/{args.space}")
        for f in files:
            print("  ", f)
        if args.dry_run:
            print((stage / "requirements.txt").read_text())
            return 0

        from huggingface_hub import HfApi

        api = HfApi()
        if args.model_revision:
            api.add_space_variable(args.space, "RECON3D_MODEL_REVISION", args.model_revision)
        info = api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(stage),
                                 commit_message=f"deploy {sha[:7]}", delete_patterns=["examples/*"])
        print(f"uploaded: {info.commit_url}")
        print("The Space now rebuilds (3-6 min). Watch the Logs tab, then open "
              f"https://{args.space.replace('/', '-').lower()}.hf.space/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
