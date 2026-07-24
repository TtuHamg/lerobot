from pathlib import Path
from pprint import pprint

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def summarize_item(item: dict) -> None:
    print("\nitem keys:")
    pprint(list(item.keys()))

    print("\nitem values:")
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            msg = f"{key}: tensor shape={tuple(value.shape)} dtype={value.dtype}"
            if value.numel() > 0 and value.is_floating_point():
                msg += f" min={value.min().item():.4f} max={value.max().item():.4f}"
            print(msg)
        else:
            print(f"{key}: {type(value)} {value}")


def main() -> None:
    root = Path(
        "/ytech_milm_intern/tujiahang/.cache/huggingface/lerobot/tuuy/lerobot_so_arm101_task0_new"
    )

    dataset = LeRobotDataset(
        repo_id="tuuy/lerobot_so_arm101_task0_new",
        root=root,
        episodes=[0],
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )

    print(dataset)
    print("root:", dataset.root)
    print("num_frames:", len(dataset))
    print("fps:", dataset.fps)

    print("\nfeatures:")
    pprint(dataset.features)

    print("\nvideo_keys:")
    pprint(dataset.meta.video_keys)

    print("\ncamera_keys:")
    pprint(dataset.meta.camera_keys)

    print("\ntasks:")
    print(dataset.meta.tasks)

    print("\nepisodes:")
    print(dataset.meta.episodes)

    index = 0

    # Stop here to inspect dataset metadata or step into dataset[index].
    breakpoint()

    item = dataset[index]
    summarize_item(item)

    # Stop here to inspect a decoded sample.
    breakpoint()


if __name__ == "__main__":
    main()
