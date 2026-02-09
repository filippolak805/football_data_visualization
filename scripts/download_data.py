from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from typing import Optional
import requests
import gdown


ROOT = Path(__file__).resolve().parents[1]  # check that imports work when running as script
sys.path.insert(0, str(ROOT))

from src.config import load_settings  # noqa: E402


CHUNK_SIZE = 1024 * 1024  # 1MB


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _extract_drive_file_id(url: str) -> Optional[str]:
    """
    Supports:
      - https://drive.google.com/file/d/<ID>/view?usp=sharing
      - https://drive.google.com/open?id=<ID>
      - https://drive.google.com/uc?id=<ID>&export=download
    """
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)

    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)

    return None


def _download_from_google_drive(file_id: str, dst: Path) -> None:
    """
    Downloads large Google Drive files by handling the confirmation token flow.
    More robust: handles cookie token OR token embedded in HTML.
    """
    _ensure_parent(dst)

    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"

    def _get_confirm_token(resp: requests.Response) -> str | None:
        # 1) cookie token
        for key, value in resp.cookies.items():
            if key.startswith("download_warning"):
                return value

        # 2) token embedded in HTML (common for large files)
        if resp.headers.get("content-type", "").startswith("text/html"):
            m = re.search(r'confirm=([0-9A-Za-z_]+)', resp.text)
            if m:
                return m.group(1)
        return None

    # First request
    resp = session.get(base_url, params={"id": file_id}, stream=False, timeout=60)
    resp.raise_for_status()

    token = _get_confirm_token(resp)

    if token:
        resp = session.get(base_url, params={"id": file_id, "confirm": token}, stream=True, timeout=60)
        resp.raise_for_status()
    else:
        resp = session.get(base_url, params={"id": file_id}, stream=True, timeout=60)
        resp.raise_for_status()  # if no token needed, re-request as stream

    content_type = resp.headers.get("content-type", "")
    if content_type.startswith("text/html"):
        head = resp.text[:500] if hasattr(resp, "text") else ""
        raise RuntimeError(
            "Google Drive download returned HTML instead of file bytes.\n"
            f"Content-Type: {content_type}\n"
            f"First 500 chars:\n{head}"
        )  # likely got an interstitial page, not the file

    total = 0
    with open(dst, "wb") as f:
        for chunk in resp.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    if dst.stat().st_size < 1024 * 100:  # <100KB certainly wrong for our dataset -> show a hint
        preview = dst.read_bytes()[:200]
        raise RuntimeError(
            f"Downloaded file is too small ({dst.stat().st_size} bytes). "
            "Likely not the dataset.\n"
            f"First 200 bytes: {preview!r}"
        )


def _download_direct(url: str, dst: Path) -> None:
    _ensure_parent(dst)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    if dst.stat().st_size == 0:
        raise RuntimeError("Downloaded file is empty. Check URL.")


def _maybe_unzip(zip_path: Path, expected_csv_path: Path) -> None:
    """
    If the download is a zip, extract it into the expected_csv_path location.
    If zip contains multiple files, we take the largest .csv.
    """
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    tmp_extract_dir = zip_path.parent / "_tmp_extract"
    tmp_extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_extract_dir)

    csv_files = list(tmp_extract_dir.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError("Zip did not contain any .csv files.")

    largest = max(csv_files, key=lambda p: p.stat().st_size)  # pick largest csv

    expected_csv_path.parent.mkdir(parents=True, exist_ok=True)
    largest.replace(expected_csv_path)

    # Cleanup
    for p in sorted(tmp_extract_dir.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink(missing_ok=True)
        else:
            p.rmdir()
    tmp_extract_dir.rmdir()


def main() -> None:
    s = load_settings()

    if not s.data_download_url:
        raise RuntimeError("DATA_DOWNLOAD_URL is empty in .env")

    if s.matches_csv_path.exists() and s.matches_csv_path.stat().st_size > 0:
        print(f"[download] OK: matches CSV already exists at {s.matches_csv_path}")
        return

    dst = s.matches_csv_path
    _ensure_parent(dst)

    # If zip mode, download to .zip then unzip to matches_csv_path
    if s.data_is_zip:
        zip_dst = dst.with_suffix(".zip")
        if zip_dst.exists() and zip_dst.stat().st_size > 0:
            print(f"[download] OK: zip already exists at {zip_dst}")
        else:
            file_id = _extract_drive_file_id(s.data_download_url)
            print(f"[download] Downloading ZIP to {zip_dst} ...")
            if file_id:
                _download_from_google_drive(file_id, zip_dst)
            else:
                _download_direct(s.data_download_url, zip_dst)
            print(f"[download] Done: {zip_dst} ({zip_dst.stat().st_size / (1024**2):.1f} MB)")

        print(f"[download] Extracting ZIP to {dst} ...")
        _maybe_unzip(zip_dst, dst)
        print(f"[download] Done: {dst} ({dst.stat().st_size / (1024**2):.1f} MB)")
        return

    # Non-zip download
    file_id = _extract_drive_file_id(s.data_download_url)
    if not file_id:
        raise RuntimeError("Could not extract Google Drive file id from DATA_DOWNLOAD_URL")

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[download] gdown downloading to {dst} ...")
    _ensure_parent(dst)
    gdown.download(url, str(dst), quiet=False)

    print(f"[download] Done: {dst} ({dst.stat().st_size / (1024**2):.1f} MB)")


if __name__ == "__main__":
    main()
