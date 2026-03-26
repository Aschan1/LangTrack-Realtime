import csv
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
MAT_PATH = REPO_ROOT / "nyu_depth_v2_labeled.mat"
OUT_DIR = REPO_ROOT / "nyu_dataset"
IMAGES_DIR = OUT_DIR / "images"
LABELS_DIR = OUT_DIR / "labels"


def decode_matlab_string(h5_file, ref) -> str:
    data = h5_file[ref][()]
    return "".join(chr(int(x)) for x in np.asarray(data).flatten())


def main() -> None:
    if not MAT_PATH.is_file():
        raise FileNotFoundError(f"Missing MAT file: {MAT_PATH}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(MAT_PATH, "r") as f:
        images = f["images"]
        labels = f["labels"]
        class_refs = f["names"][0]
        scene_refs = f["scenes"][0]
        scene_type_refs = f["sceneTypes"][0]
        raw_rgb_refs = f["rawRgbFilenames"][0]

        class_names = {"0": "unlabeled"}
        for idx, ref in enumerate(class_refs, start=1):
            class_names[str(idx)] = decode_matlab_string(f, ref)

        with (OUT_DIR / "class_names.json").open("w", encoding="utf-8") as handle:
            json.dump(class_names, handle, indent=2, ensure_ascii=False)

        with (OUT_DIR / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["index", "image_path", "label_path", "scene", "scene_type", "raw_rgb_filename"],
            )
            writer.writeheader()

            total = images.shape[0]
            for index in tqdm(range(total), desc="Exporting NYU dataset"):
                image = np.transpose(images[index], (2, 1, 0))
                label = np.transpose(labels[index], (1, 0)).astype(np.uint16)

                image_name = f"{index:04d}.png"
                label_name = f"{index:04d}.png"

                Image.fromarray(image).save(IMAGES_DIR / image_name)
                Image.fromarray(label).save(LABELS_DIR / label_name)

                writer.writerow(
                    {
                        "index": index,
                        "image_path": f"images/{image_name}",
                        "label_path": f"labels/{label_name}",
                        "scene": decode_matlab_string(f, scene_refs[index]),
                        "scene_type": decode_matlab_string(f, scene_type_refs[index]),
                        "raw_rgb_filename": decode_matlab_string(f, raw_rgb_refs[index]),
                    }
                )


if __name__ == "__main__":
    main()
