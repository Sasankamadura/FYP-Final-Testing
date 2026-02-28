"""Fix decoder_layers in all saved result JSON files.

Only these 3 models have 6 decoder layers:
  - gnconv_p2
  - gnconv_p2_repvgg
  - gnconv_slim_p2_repvgg

All other models have 3 decoder layers.
"""
import json
import os
import glob

SIX_DECODER = {"gnconv_p2", "gnconv_p2_repvgg", "gnconv_slim_p2_repvgg"}

def fix_combined_json(path):
    """Fix a combined JSON file (all_validation_results.json or benchmark_results.json)."""
    if not os.path.exists(path):
        print(f"  [SKIP] Not found: {path}")
        return
    with open(path, "r") as f:
        data = json.load(f)
    changed = False
    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        cfg = val.get("config", val)
        if "decoder_layers" in cfg:
            correct = 6 if key in SIX_DECODER else 3
            if cfg["decoder_layers"] != correct:
                old = cfg["decoder_layers"]
                cfg["decoder_layers"] = correct
                changed = True
                print(f"  {os.path.basename(path)}: {key}  {old} -> {correct}")
    if changed:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  SAVED: {path}")
    else:
        print(f"  {os.path.basename(path)}: no changes needed")


def fix_individual_json(path):
    """Fix an individual model result JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    key = os.path.basename(path).replace("_results.json", "")
    if "config" in data and "decoder_layers" in data["config"]:
        correct = 6 if key in SIX_DECODER else 3
        if data["config"]["decoder_layers"] != correct:
            old = data["config"]["decoder_layers"]
            data["config"]["decoder_layers"] = correct
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  {os.path.basename(path)}: {old} -> {correct}")


if __name__ == "__main__":
    base = os.path.join("eval", "results", "GPU_NVIDIA_GeForce_RTX_4050_Laptop_GPU")

    print("Fixing combined validation results...")
    fix_combined_json(os.path.join(base, "validation", "all_validation_results.json"))

    print("Fixing combined benchmark results...")
    fix_combined_json(os.path.join(base, "benchmark", "benchmark_results.json"))

    print("Fixing individual validation result files...")
    for f in sorted(glob.glob(os.path.join(base, "validation", "*_results.json"))):
        if "all_" not in os.path.basename(f):
            fix_individual_json(f)

    print("\nDone! All decoder_layers values corrected.")
