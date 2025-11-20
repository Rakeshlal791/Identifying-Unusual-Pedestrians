import os
import json
from torch.utils.data import Dataset
from dataset import parse_occlusion

# Keypoint name normalization map
NAME_MAP = {
    "left_eye": "eye_left",
    "right_eye": "eye_right",
    "left_ear": "ear_left",
    "right_ear": "ear_right",
    "left_shoulder": "shoulder_left",
    "right_shoulder": "shoulder_right",
    "left_elbow": "elbow_left",
    "right_elbow": "elbow_right",
    "left_wrist": "wrist_left",
    "right_wrist": "wrist_right",
    "left_hip": "hip_left",
    "right_hip": "hip_right",
    "left_knee": "knee_left",
    "right_knee": "knee_right",
    "left_ankle": "ankle_left",
    "right_ankle": "ankle_right",
}

class SyntheticCombinedTestDataset(Dataset):
    """
    Combines:
      - Synthetic annotations from syn_images/
      - Normal EUROCITYPERSON-style annotations from normal_images/
    Returns uniform dict:
        {
            "bbox": [x0, y0, x1, y1],
            "keypoints": {name: {"x": val, "y": val}},
            "frame_path": <annotation file path>,
            "synthetic": 0 or 1
        }
    """

    def __init__(self, root_synthetic, root_normal):
        self.instances = []

        # ------------------------------------
        # 1️⃣ Load synthetic annotations
        # ------------------------------------
        syn_ann_root = os.path.join(root_synthetic, ".annotations_aug")
        if not os.path.isdir(syn_ann_root):
            alt = os.path.join(root_synthetic, "annotations")
            syn_ann_root = alt if os.path.isdir(alt) else None

        if syn_ann_root:
            for fname in sorted(os.listdir(syn_ann_root)):
                if not fname.endswith(".json"):
                    continue
                ann_path = os.path.join(syn_ann_root, fname)
                with open(ann_path, "r") as f:
                    data = json.load(f)

                for ped in data.get("pedestrians", []):
                    bbox = ped["bbox"]
                    x0, y0 = bbox["x"], bbox["y"]
                    x1, y1 = x0 + bbox["w"], y0 + bbox["h"]

                    keypoints = {}
                    for kp in ped["keypoints"]:
                        name = NAME_MAP.get(kp["name"], kp["name"])
                        keypoints[name] = {"x": kp["x"], "y": kp["y"]}

                    self.instances.append({
                        "frame_path": ann_path,  # just identifier
                        "bbox": [x0, y0, x1, y1],
                        "keypoints": keypoints,
                        "synthetic": 1,
                        "id": ped.get("id"),
                    })

        # ------------------------------------
        # 2️⃣ Load normal EUROCITYPERSON-style annotations
        # ------------------------------------
        for ann_file in sorted(os.listdir(root_normal)):
            if not ann_file.endswith(".json"):
                continue

            ann_path = os.path.join(root_normal, ann_file)
            with open(ann_path, "r") as f:
                data = json.load(f)

            W, H = data.get("imagewidth"), data.get("imageheight")

            for child in data.get("children", []):
                if child.get("identity") != "pedestrian":
                    continue
                if not child.get("children"):
                    continue

                occ_val = parse_occlusion(child.get("tags", []))
                if occ_val is not None and occ_val > 10:
                    continue

                skels = [c for c in child["children"] if c.get("identity") == "skeleton"]
                if not skels:
                    continue
                keypoints = skels[0]["joints"]

                x0, y0, x1, y1 = child["x0"], child["y0"], child["x1"], child["y1"]
                if x1 - x0 < 10 or y1 - y0 < 10:
                    continue
                if W and H and (x0 < 0 or y0 < 0 or x1 > W or y1 > H):
                    continue

                self.instances.append({
                    "frame_path": ann_path,  # use JSON file path as ID
                    "bbox": [x0, y0, x1, y1],
                    "keypoints": keypoints,
                    "synthetic": 0,
                    "id": None,
                })

        print(f"Loaded {len(self.instances)} total instances "
              f"({sum(i['synthetic']==1 for i in self.instances)} synthetic, "
              f"{sum(i['synthetic']==0 for i in self.instances)} normal)")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        entry = self.instances[idx]
        target = {
            "bbox": entry["bbox"],
            "keypoints": entry["keypoints"],
            "frame_path": entry["frame_path"],
            "synthetic": entry["synthetic"],
            "id": entry.get("id"),
        }
        return entry["frame_path"], target, False
