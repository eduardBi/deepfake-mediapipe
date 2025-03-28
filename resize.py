import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Directory to save cropped faces
output_dir = "cropped_faces"
os.makedirs(output_dir, exist_ok=True)

# Desired face size (width x height)
FACE_SIZE = (512, 512)

# Face expansion factor (increases bounding box size evenly)
EXPAND_RATIO = 0.3  # 30% expansion in both width & height

# Process images from 333.png to 344.png
for i in range(333, 349):
    image_path = f"{i}.png"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image '{image_path}'")
        continue  # Skip missing images

    # Convert to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
        results = face_detection.process(rgb_image)

        # If no face detected, skip
        if not results.detections:
            print(f"No face found in '{image_path}'")
            continue

        for idx, detection in enumerate(results.detections):
            bbox = detection.location_data.relative_bounding_box

            # Convert bbox from relative to absolute coordinates
            h, w, _ = image.shape
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            x_max = int((bbox.xmin + bbox.width) * w)
            y_max = int((bbox.ymin + bbox.height) * h)

            # Calculate face width & height
            face_width = x_max - x_min
            face_height = y_max - y_min

            # Expand bounding box symmetrically to maintain proportions
            expand_w = int(face_width * EXPAND_RATIO)
            expand_h = int(face_height * EXPAND_RATIO)

            x_min = max(0, x_min - expand_w)
            x_max = min(w, x_max + expand_w)
            y_min = max(0, y_min - expand_h)
            y_max = min(h, y_max + expand_h)

            # Ensure bounding box is square
            box_width = x_max - x_min
            box_height = y_max - y_min
            side_length = max(box_width, box_height)

            # Center the square bounding box
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            x_min = max(0, center_x - side_length // 2)
            x_max = min(w, center_x + side_length // 2)
            y_min = max(0, center_y - side_length // 2)
            y_max = min(h, center_y + side_length // 2)

            # Crop the face while maintaining aspect ratio
            cropped_face = image[y_min:y_max, x_min:x_max]

            # Resize while keeping proportions (add padding if needed)
            resized_face = cv2.resize(cropped_face, (FACE_SIZE[0], FACE_SIZE[1]), interpolation=cv2.INTER_AREA)

            # Save cropped face
            output_path = os.path.join(output_dir, f"face_{i}_{idx}.png")
            cv2.imwrite(output_path, resized_face)

            # Show the cropped face (optional)
            cv2.imshow(f"Face {i}-{idx}", resized_face)

    print(f"Processed '{image_path}'")

# Keep windows open until 'q' is pressed
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
