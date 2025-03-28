import cv2
import signal
import sys
from deepface import DeepFace

# Path to the image
image_path = "/home/ed/projects/faceswap/FaceFusion/338.png"  # Change this to your image path

# Load the image
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image. Check the file path.")
    sys.exit(1)

# Graceful exit on CTRL+C
def signal_handler(sig, frame):
    print("\n[INFO] Process interrupted. Exiting gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)  # Handle CTRL+C

try:
    # Analyze the image for emotions, gender, and race (excluding age)
    result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)

    if isinstance(result, list):
        result = result[0]  # Extract first result if it's a list

    # Print all extracted DeepFace data (excluding age)
    print("[DeepFace Analysis Result]:", result)

    # Extract the dominant emotion
    emotion = result.get('dominant_emotion', 'Unknown')

    # Display the detected emotion on the image
    cv2.putText(image, f"Emotion: {emotion}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the image with detected emotion
    cv2.imshow('Emotion Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print("Error:", e)
