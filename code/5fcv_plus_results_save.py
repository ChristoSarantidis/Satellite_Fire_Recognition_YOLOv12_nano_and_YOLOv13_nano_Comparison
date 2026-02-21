import os
import time
import pandas as pd
from ultralytics import YOLO

# Set your parameters
folds = ['folds_normal/fold1.yaml','folds_normal/fold2.yaml', 'folds_normal/fold3.yaml', 'folds_normal/fold4.yaml', 'folds_normal/fold5.yaml']

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

##########################################################################################################################yolov13

from ultralytics.nn.modules.block import DSC3k2
model_path = 'yolov13n.pt'  # Change to yolov10n.pt, etc.


# Store results
fold_results = []

for fold_yaml in folds:
    fold_name = os.path.splitext(os.path.basename(fold_yaml))[0]
    print(f"Training on {fold_name}...")

    # Load model
    model = YOLO(model_path)

    # --- Training ---
    start_train = time.time()
    model.train(data=fold_yaml, epochs=130, imgsz=1000, project='runs_crossval', name=fold_name, device=[0,1,2], batch=39, save_txt=True, save=True, augment=True, optimizer='SGD')
    #model.train(data=fold_yaml, epochs=140, imgsz=1024, device="0", workers=0,batch=9, amp=True, project="/home/sarantidis/Book_Publication/runs_crossval", name=fold_name,exist_ok=True,)
    train_time = time.time() - start_train

    # --- Validation ---
    start_val = time.time()
    metrics = model.val(data=fold_yaml)
    val_time = time.time() - start_val

#print the metrics before extracting###################################################################
    print("***********************************************************")
    print("model", os.path.basename(model_path))
    print("fold", fold_name)
    print("p", metrics.box.mp)
    print("r", metrics.box.mr)
    print("f1", metrics.box.f1)
    print("sum_f1", sum(metrics.box.f1))
    print("len_f1", len(metrics.box.f1))
    print("map50", metrics.box.map50)
    print("train_time", train_time)
    print("val_time", val_time)
    print("***********************************************************")
##########################################################################################################
    
    # Extract metrics
    result = {
        'model': os.path.basename(model_path),
        'fold': fold_name,
        'precision': round(metrics.box.mp, 3),         # ✅ no parentheses
        'recall': round(metrics.box.mr, 3),            # ✅
        'f1': round(sum(metrics.box.f1)/len(metrics.box.f1), 3),  # still valid
        'map50': round(metrics.box.map50, 3),          # ✅
        'train_time_sec': round(train_time, 2),
        'val_time_sec': round(val_time, 2)
    }

    fold_results.append(result)


    #IF FOLD IS NOT FOLD 5 THEN SLEEP FOR 5*60 SECONDS################################################
    
    if fold_name != 'fold5':
        print("sleeping")
        time.sleep(2*60)
        print("awake")
    
    # Pause between folds to manage resources

# Save results to CSV
df = pd.DataFrame(fold_results)
csv_filename = os.path.join(results_dir, f'{os.path.splitext(os.path.basename(model_path))[0]}_crossval_results.csv')
df.to_csv(csv_filename, index=False)

print(f"\n✅ Results saved to: {csv_filename}")
print(df)

