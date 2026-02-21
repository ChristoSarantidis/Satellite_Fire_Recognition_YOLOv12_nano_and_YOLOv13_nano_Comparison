import os
import cv2

# Step 1: Set the path
input_folder = 'E:/Book_Publication/Upload_to_server3/images'  # Change this to your folder name
output_folder = os.path.join(input_folder, 'heq')

# Step 2: Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Step 3: Process each image
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not read {filename}")
            continue

        # Check if image is grayscale or color
        if len(img.shape) == 2 or img.shape[2] == 1:
            # Grayscale image
            equalized = cv2.equalizeHist(img)
        else:
            # Color image – apply HEQ to each channel separately
            channels = cv2.split(img)
            eq_channels = [cv2.equalizeHist(ch) for ch in channels]
            equalized = cv2.merge(eq_channels)

        # Save the equalized image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, equalized)
        print(f"Saved: {output_path}")
    else:
        print('Filetype not found')