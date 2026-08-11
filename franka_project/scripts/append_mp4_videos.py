#!/usr/bin/env python3
"""Join MP4 files from a folder, optionally after a base video."""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def ffmpeg_list_entry(path: Path) -> str:
    """Format a path for an ffmpeg concat-list file."""
    escaped = path.resolve().as_posix().replace("'", "'\\''")
    return f"file '{escaped}'\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join MP4 files from a folder in filename order."
    )
    parser.add_argument("input_folder", type=Path, help="Folder containing MP4 files")
    parser.add_argument("output_video", type=Path, help="Output MP4 path")
    parser.add_argument(
        "--base-video",
        type=Path,
        help="Optional video placed before the folder videos",
    )
    args = parser.parse_args()

    if args.base_video is not None and not args.base_video.is_file():
        parser.error(f"Base video does not exist: {args.base_video}")
    if not args.input_folder.is_dir():
        parser.error(f"Input folder does not exist: {args.input_folder}")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg was not found. Please install ffmpeg first.")

    base_video = args.base_video.resolve() if args.base_video is not None else None
    output_video = args.output_video.resolve()
    folder_videos = sorted(
        (
            path.resolve()
            for path in args.input_folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".mp4"
        ),
        key=lambda path: path.name.lower(),
    )
    excluded_paths = {output_video}
    if base_video is not None:
        excluded_paths.add(base_video)
    folder_videos = [path for path in folder_videos if path not in excluded_paths]

    if not folder_videos:
        parser.error(f"No MP4 files found in: {args.input_folder}")

    videos = ([base_video] if base_video is not None else []) + folder_videos
    output_video.parent.mkdir(parents=True, exist_ok=True)

    print("Videos will be joined in this order:")
    for index, video in enumerate(videos, start=1):
        print(f"  {index}. {video}")
    print(flush=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", encoding="utf-8"
    ) as concat_file:
        concat_file.writelines(ffmpeg_list_entry(video) for video in videos)
        concat_file.flush()

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file.name,
            "-c",
            "copy",
            str(output_video),
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                "Video concatenation failed. Make sure the videos use compatible "
                "codec, resolution, frame rate, and audio settings."
            ) from exc

    print(f"Done: {output_video}")


if __name__ == "__main__":
    main()
