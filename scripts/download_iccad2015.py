#!/usr/bin/env python3

import argparse
import html.parser
import os
import pathlib
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"

DATASETS = {
    "ot": {
        "file_id": "1xeauwLR9lOxnYvsK2JGPSY0INQh8VuE4",
        "archive_name": "iccad2015.ot.tar.gz",
        "target_dir": BENCHMARKS_DIR / "iccad2015.ot",
    },
    "hs": {
        "file_id": "1HsAW_qcRje_-Ex1anWqAEQOKpGeCxpZa",
        "archive_name": "iccad2015.hs.tar.gz",
        "target_dir": BENCHMARKS_DIR / "iccad2015.hs",
    },
}


class _DriveDownloadFormParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self._in_form = False
        self.action = None
        self.inputs = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self._in_form = True
            self.action = attrs.get("action")
            return

        if self._in_form and tag == "input":
            name = attrs.get("name")
            value = attrs.get("value", "")
            if name:
                self.inputs[name] = value

    def handle_endtag(self, tag):
        if tag == "form" and self._in_form:
            self._in_form = False


def _download_url(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _open_url(url, headers=None):
    request_headers = {
        "User-Agent": "Mozilla/5.0",
    }
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(
        url,
        headers=request_headers,
    )
    return urllib.request.urlopen(request, timeout=120)


def _resolve_drive_response(file_id):
    with _open_url(_download_url(file_id)) as response:
        content_type = response.headers.get("content-type", "")
        final_url = response.geturl()

        if "text/html" not in content_type:
            return final_url, None

        html_text = response.read().decode("utf-8", "ignore")
        parser = _DriveDownloadFormParser()
        parser.feed(html_text)

    if not parser.action or not parser.inputs:
        raise RuntimeError(
            "Failed to parse Google Drive confirmation page. "
            "The file may require manual download from the browser."
        )

    query = urllib.parse.urlencode(parser.inputs)
    return f"{parser.action}?{query}", parser.inputs


def _download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_size = destination.stat().st_size if destination.exists() else 0
    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    with _open_url(url, headers=headers) as response:
        total = response.headers.get("content-length")
        total = int(total) if total and total.isdigit() else None

        if existing_size > 0 and getattr(response, "status", None) == 206:
            mode = "ab"
            downloaded = existing_size
            total = existing_size + total if total is not None else None
        else:
            mode = "wb"
            downloaded = 0

        with open(destination, mode) as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = downloaded * 100.0 / total
                    print(f"downloaded {downloaded}/{total} bytes ({percent:.1f}%)", flush=True)
                else:
                    print(f"downloaded {downloaded} bytes", flush=True)


def _extract_archive(archive_path, target_dir):
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="iccad2015_extract_") as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(temp_path)

        entries = list(temp_path.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            extracted_root = entries[0]
        else:
            extracted_root = temp_path

        shutil.move(str(extracted_root), str(target_dir))


def _dataset_choices(dataset):
    if dataset == "all":
        return ["ot", "hs"]
    return [dataset]


def main():
    parser = argparse.ArgumentParser(description="Download ICCAD2015 benchmark tarballs from Google Drive.")
    parser.add_argument("dataset", choices=["ot", "hs", "all"], help="Benchmark set to download")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the downloaded tar.gz file")
    args = parser.parse_args()

    for dataset in _dataset_choices(args.dataset):
        config = DATASETS[dataset]
        archive_path = BENCHMARKS_DIR / config["archive_name"]
        target_dir = config["target_dir"]

        print(f"resolving Google Drive link for {dataset} ...", flush=True)
        download_url, _ = _resolve_drive_response(config["file_id"])

        print(f"downloading {dataset} archive to {archive_path} ...", flush=True)
        _download_file(download_url, archive_path)

        print(f"extracting {archive_path} to {target_dir} ...", flush=True)
        _extract_archive(archive_path, target_dir)

        if not args.keep_archive and archive_path.exists():
            archive_path.unlink()

        print(f"{dataset} benchmark ready at {target_dir}", flush=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"download failed: {error}", file=sys.stderr)
        raise SystemExit(1)
