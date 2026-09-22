import json
import os
import random
import glob

def generate_identity_split(manifest_dir: str, train_ratio=0.6, val_ratio=0.2):
    """
    Generates a strict identity-aware split to prevent data leakage.
    Ensures an identity belongs entirely to Train, Val, or Test.
    """
    manifest_files = glob.glob(os.path.join(manifest_dir, "*_manifest.json"))
    
    identities = set()
    identity_map = {}
    
    for mf in manifest_files:
        with open(mf, "r") as f:
            data = json.load(f)
            iid = data["identity_id"]
            identities.add(iid)
            if iid not in identity_map:
                identity_map[iid] = []
            identity_map[iid].append(data)
            
    identities = sorted(list(identities))
    
    if len(identities) < 5:
        print(f"BLOCKED: Only {len(identities)} identities found. Multi-identity data must be acquired (Target >= 5).")
        return None
        
    # Shuffle for randomness in selection, but seeded for reproducibility if desired
    random.seed(42)
    random.shuffle(identities)
    
    n_total = len(identities)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    
    train_ids = set(identities[:n_train])
    val_ids = set(identities[n_train:n_train+n_val])
    test_ids = set(identities[n_train+n_val:])
    
    split = {
        "train_identities": list(train_ids),
        "val_identities": list(val_ids),
        "test_identities": list(test_ids),
        "leakage_check": {
            "cross_split_leakage": bool(train_ids.intersection(val_ids) or train_ids.intersection(test_ids) or val_ids.intersection(test_ids))
        },
        "sequence_counts": {
            "train": sum(len(identity_map[i]) for i in train_ids),
            "val": sum(len(identity_map[i]) for i in val_ids),
            "test": sum(len(identity_map[i]) for i in test_ids)
        }
    }
    
    print("Identity-Aware Split Generated successfully.")
    print(json.dumps(split, indent=2))
    return split

if __name__ == "__main__":
    generate_identity_split("../../synthesia_training_data/manifests")
