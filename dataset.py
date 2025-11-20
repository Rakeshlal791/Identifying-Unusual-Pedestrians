import os
import json
import random
from torch.utils.data import Dataset


def find_image(img_dir, base_name):
    """Find image file matching base_name in img_dir, regardless of extension/case."""
    for fname in os.listdir(img_dir):
        name, ext = os.path.splitext(fname)
        if name.lower() == base_name.lower() and ext.lower() in [".jpg", ".jpeg", ".png"]:
            return os.path.join(img_dir, fname)
    return None


def parse_occlusion(tags):
    """Parse occlusion value from tags like ['occluded>10'] → int or None."""
    if not tags:
        return None
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("occluded>"):
            try:
                return int(tag.split(">")[1])
            except Exception:
                return None
    return None


# ✅ Define left–right joint pairs for semantic swapping
LEFT_RIGHT_PAIRS = [
    ("eye_left", "eye_right"),
    ("ear_left", "ear_right"),
    ("shoulder_left", "shoulder_right"),
    ("elbow_left", "elbow_right"),
    ("wrist_left", "wrist_right"),
    ("hip_left", "hip_right"),
    ("knee_left", "knee_right"),
    ("ankle_left", "ankle_right"),
]


class PedestrianInstanceDataset(Dataset):
    """
    Instance-centric dataset (keypoints-only):
      - Does NOT open image files
      - Returns the image path instead of an image tensor
    """

    def __init__(
        self,
        root_annotations,
        root_images,
        split="train",
        occlusion_threshold=10,
        min_box_area=400,
    ):
        self.root_ann = os.path.join(root_annotations, split)
        self.root_img = os.path.join(root_images, split)
        self.split = split
        self.occlusion_threshold = occlusion_threshold
        self.min_box_area = min_box_area

        self.instances = []

        if not os.path.isdir(self.root_ann):
            raise FileNotFoundError(f"Annotations split folder not found: {self.root_ann}")
        if not os.path.isdir(self.root_img):
            raise FileNotFoundError(f"Images split folder not found: {self.root_img}")

        frame_idx = 0
        for city in sorted(os.listdir(self.root_ann)):
            city_ann_dir = os.path.join(self.root_ann, city, "front")
            if not os.path.isdir(city_ann_dir):
                continue

            img_dir = os.path.join(self.root_img, city, "front")
            if not os.path.isdir(img_dir):
                continue

            for ann_file in sorted(os.listdir(city_ann_dir)):
                if not ann_file.endswith(".json"):
                    continue

                base_name = os.path.splitext(ann_file)[0]
                ann_path = os.path.join(city_ann_dir, ann_file)
                img_path = find_image(img_dir, base_name)
                if img_path is None:
                    continue

                with open(ann_path, "r") as f:
                    data = json.load(f)

                W, H = data.get("imagewidth", None), data.get("imageheight", None)

                ped_idx = 0
                for child in data.get("children", []):
                    if child.get("identity") != "pedestrian":
                        continue
                    if not child.get("children"):
                        continue

                    # --- occlusion filter ---
                    occ_val = parse_occlusion(child.get("tags", []))
                    if occ_val is not None and occ_val > self.occlusion_threshold:
                        continue

                    # --- skeleton check ---
                    skels = [c for c in child["children"] if c.get("identity") == "skeleton"]
                    if not skels:
                        continue

                    # --- bbox filtering ---
                    bbox = [child["x0"], child["y0"], child["x1"], child["y1"]]
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    area = w * h
                    if area < self.min_box_area:
                        continue

                    # --- full in-frame check ---
                    if W is not None and H is not None:
                        x0, y0, x1, y1 = bbox
                        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
                            continue  # skip truncated pedestrians

                    keypoints = skels[0]["joints"]

                    self.instances.append({
                        "frame_path": img_path,   # just the path
                        "frame_idx": frame_idx,
                        "ped_idx": ped_idx,
                        "bbox": bbox,
                        "keypoints": keypoints,
                        "width": W,
                        "height": H
                    })
                    ped_idx += 1

                frame_idx += 1

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        entry = self.instances[idx]

        bbox = entry["bbox"].copy()
        keypoints = {k: v.copy() for k, v in entry["keypoints"].items()}
        W = entry["width"]

        target = {
            "bbox": bbox,
            "keypoints": keypoints,
            "frame_path": entry["frame_path"],
            "frame_idx": entry["frame_idx"],
            "ped_idx": entry["ped_idx"],
        }

        # Return image path and flipped flag
        return entry["frame_path"], target
