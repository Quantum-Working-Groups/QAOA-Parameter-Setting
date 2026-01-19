import json
import numpy as np
from sklearn.cluster import DBSCAN


# ---------- helpers ----------

def parse_key(k: str):
    parts = [p.strip() for p in k.split(",")]
    if len(parts) < 2:
        raise ValueError(f"Bad key format: {k!r}")
    a = int(parts[0])
    b = float(parts[1])
    c = parts[2] if len(parts) >= 3 else None
    return a, b, c


def extract_qaoa_angles_and_metadata(v):
    """
    Returns:
        qaoa_angles, metadata_dict
    """
    if isinstance(v, dict):
        if "qaoa_angles" in v:
            qaoa_angles = v["qaoa_angles"]
            metadata = {k: val for k, val in v.items() if k != "qaoa_angles"}
            return qaoa_angles, metadata
        else:
            return None, dict(v)
    else:
        return v, {}


def to_list(x):
    """
    Normalize qaoa_angles to a list so we can flatten ONE level safely.
    """
    if x is None:
        return []

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, dict):
        try:
            keys = sorted(x.keys(), key=lambda k: int(k))
        except Exception:
            keys = sorted(x.keys())
        return [x[k] for k in keys]

    if isinstance(x, (list, tuple)):
        return list(x)

    return [x]


# ---------- load database ----------

database_path = "./optimized_angles_database_L2F.json"
with open(database_path, "r", encoding="utf-8") as f:
    db = json.load(f)


# ---------- group by first index (a) ----------

groups = {}
for k, v in db.items():
    a, b, c = parse_key(k)
    groups.setdefault(a, []).append((k, b, c, v))


# ---------- clustering params ----------

eps = 0.1
min_samples = 2
keep_noise = True
center = "median"  # or "mean"


# ---------- clustering ----------

new_db = {}

for a, items in groups.items():
    bs = np.array([it[1] for it in items], dtype=float).reshape(-1, 1)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(bs)

    lab2idx = {}
    for i, lab in enumerate(labels):
        if lab == -1 and not keep_noise:
            continue
        lab2idx.setdefault(int(lab), []).append(i)

    for lab, idxs in lab2idx.items():

        # ----- noise: cluster of size 1 -----
        if lab == -1:
            for i in idxs:
                orig_k, b, c, v = items[i]
                qaoa_angles_i, meta_from_v = extract_qaoa_angles_and_metadata(v)

                center_key = f"{a}, {float(b)}"

                new_db[center_key] = {
                    "metadata": {
                        "cluster_type": "noise",
                        "a": a,
                        "center_b": float(b),
                        "size": 1,
                        "members": [
                            {"orig_key": orig_k, "b": float(b), "c": c}
                        ],
                        **meta_from_v,
                    },
                    "qaoa_angles": to_list(qaoa_angles_i),
                }
            continue

        # ----- normal cluster -----
        b_vals = bs[idxs, 0]
        center_b = float(np.mean(b_vals)) if center == "mean" else float(np.median(b_vals))
        center_key = f"{a}, {center_b}"

        members = []
        qaoa_angles_flat = []
        member_metadata = []

        for i in idxs:
            orig_k, b, c, v = items[i]
            qaoa_angles_i, meta_i = extract_qaoa_angles_and_metadata(v)

            members.append({"orig_key": orig_k, "b": float(b), "c": c})

            # 🔑 AQUÍ SE ELIMINA EL NIVEL EXTRA
            qaoa_angles_flat.extend(to_list(qaoa_angles_i))

            if meta_i:
                member_metadata.append(meta_i)

        new_db[center_key] = {
            "metadata": {
                "cluster_type": "cluster",
                "a": a,
                "center_b": center_b,
                "size": len(idxs),
                "members": members,
                "member_metadata": member_metadata,
            },
            "qaoa_angles": qaoa_angles_flat,
        }


# ---------- save ----------

with open("optimized_angles_database_L2F_cluster.json", "w", encoding="utf-8") as f:
    json.dump(new_db, f, indent=2, ensure_ascii=False)
