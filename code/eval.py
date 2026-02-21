import os
import yaml
import time
import pandas as pd
from ultralytics import YOLO

# === CONFIGURATION ===
#BASE_DIR = "/home/sarantidis/Book_Publication/b_v12/baseline/120_epochs_error/runs_crossval"  # where trained model folders are
BASE_DIR= "/home/sarantidis/Book_Publication/a_final_experiments/v13/augment/runs_crossval"
DATASET_DIR = "/home/sarantidis/Book_Publication/folds_normal/"  # where fold YAMLs and datasets are


################################################################################################yolov13
OUTPUT_CSV = "v13.csv"
TARGET_ALGORITHM = "yolov13"  # change this to yolov10, yolov12, etc.

# === Collect matching fold directories (e.g., yolov9_fold1, yolov9_fold2, ...)
model_folders = sorted([
    f for f in os.listdir(BASE_DIR)
    if f.startswith(TARGET_ALGORITHM) and "fold" in f
])

results = []

# === Process each fold
for model_folder in model_folders:
    print(f"\n🚀 Evaluating test set for {model_folder}")
    fold_num = model_folder.split("fold")[-1]
    model_path = os.path.join(BASE_DIR, model_folder, "weights", "best.pt")
    model = YOLO(model_path)

    # Use the same foldX.yaml (it already contains test:)
    fold_yaml_path = os.path.join(DATASET_DIR, f"fold{fold_num}.yaml")
    if not os.path.exists(fold_yaml_path):
        print(f"❌ Missing YAML: {fold_yaml_path}")
        continue

    # Run test evaluation
    start_test = time.time()
    test_metrics = model.val(data=fold_yaml_path, split='test', verbose=False)
    test_time_sec = round(time.time() - start_test, 2)

    # Collect metrics
    results.append({
        "algorithm": TARGET_ALGORITHM,
        "fold": fold_num,
        "precision": round(test_metrics.box.mp, 3),
        "recall": round(test_metrics.box.mr, 3),
        "f1": round(sum(test_metrics.box.f1)/len(test_metrics.box.f1), 3) if test_metrics.box.f1 else 0.0,
        "map50": round(test_metrics.box.map50, 3),
        "map50_95": round(test_metrics.box.map, 3),
        "test_time_sec": test_time_sec
    })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Test results saved to {OUTPUT_CSV}")
print(df)