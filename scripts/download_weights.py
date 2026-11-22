"""Download the weights listed in ``vvm/weights.py``.

    python scripts/download_weights.py                 # everything
    python scripts/download_weights.py --model gem-x   # one model
    python scripts/download_weights.py --check         # status only
    python scripts/download_weights.py --token hf_xxx  # token for gated Hugging Face repos

Licence-gated SMPL-family files are not downloaded; the script prints where to put them.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vvm.paths import CHECKPOINTS  # noqa: E402
from vvm.weights import WEIGHTS, Weight  # noqa: E402


def _fetch_url(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "video-vs-mocap-downloader"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(out)


def _download(w: Weight) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    dest = w.path
    # Ungated repos are fetched anonymously so that a stale stored token cannot break them.
    token = None if w.gated else False
    if w.source == "hf":
        local = hf_hub_download(w.repo, w.filename, revision=w.revision, token=token)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
    elif w.source == "hf_snapshot":
        snapshot_download(w.repo, revision=w.revision, local_dir=dest, token=token,
                          allow_patterns=list(w.allow_patterns) or None)
    elif w.source == "url":
        _fetch_url(w.repo, dest)
    elif w.source == "url_zip_member":
        with tempfile.TemporaryDirectory() as td:
            z = Path(td) / "a.zip"
            _fetch_url(w.repo, z)
            with zipfile.ZipFile(z) as archive:
                member = next(n for n in archive.namelist() if n.endswith(w.filename))
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    elif w.source == "github_tar":
        with tempfile.TemporaryDirectory() as td:
            t = Path(td) / "a.tar.gz"
            _fetch_url(w.repo, t)
            with tarfile.open(t) as tar:
                tar.extractall(dest.parent, filter="data")
    elif w.source in ("gdrive_file", "gdrive_folder"):
        exe = shutil.which("gdown") or str(Path(sys.executable).parent / "gdown")
        if w.source == "gdrive_file":
            is_dir = "directory" in w.note  # keep the Drive file name inside dest/
            (dest if is_dir else dest.parent).mkdir(parents=True, exist_ok=True)
            out = str(dest) + "/" if is_dir else str(dest)
            cmd = [exe, "-O", out, f"https://drive.google.com/uc?id={w.repo}"]
        else:
            dest.mkdir(parents=True, exist_ok=True)
            cmd = [exe, "--folder", "-O", str(dest), f"https://drive.google.com/drive/folders/{w.repo}"]
        subprocess.run(cmd, check=True)
    else:
        raise ValueError(f"cannot download source={w.source}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="only weights used by this model key")
    ap.add_argument("--only", nargs="*", help="only these manifest keys")
    ap.add_argument("--check", action="store_true", help="report status without downloading")
    ap.add_argument("--token", help="Hugging Face access token for gated repos "
                                    "(default: $HF_TOKEN or the token saved by `hf auth login`)")
    args = ap.parse_args()
    if args.token:  # exported for huggingface_hub only, never printed
        os.environ["HF_TOKEN"] = args.token.strip()

    todo = [w for w in WEIGHTS
            if (not args.model or args.model in w.models) and (not args.only or w.key in args.only)]
    failed, manual = [], []
    for w in todo:
        rel = w.path.relative_to(CHECKPOINTS.parent)
        if w.exists():
            print(f"[ok]      {rel}")
            continue
        if w.source == "manual":
            manual.append(w)
            print(f"[manual]  {rel}  <- {w.note}")
            continue
        if args.check:
            print(f"[missing] {rel}{'  (gated)' if w.gated else ''}")
            continue
        print(f"[get]     {rel} ...", flush=True)
        try:
            _download(w)
        except Exception as e:  # keep going; report at the end
            failed.append((w, e))
            print(f"[fail]    {rel}: {type(e).__name__}: {str(e)[:300]}")
    if failed:
        print("\nFailed downloads:")
        for w, e in failed:
            hint = " (gated: request access on huggingface.co and run `hf auth login`)" if w.gated else ""
            print(f"  {w.key}: {type(e).__name__}{hint}")
    if manual:
        print("\nLicence-gated body models must be placed by hand; see README > Installation > Model weights.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
