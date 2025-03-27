import cv2
import mediapipe as mp

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define facial landmarks based on user selection
FACE_PARTS = {
    "Nose": [1, 8, 236, 456],
    "Lips": [0, 37, 267,  87, 14, 317, 84, 17, 314,82,13,312  ,61 ,291],
    "Chin": [148, 152, 377],
    "Eyes": [113,27,190,342,257,414,23,253],
    "Head": [109, 10, 338, 116, 345],
    "lashes":[46,55 ,52,285,282,276]
}

# Load image
image_path = "55.png"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not read image at '{image_path}'")
    exit()

# Convert to RGB (MediaPipe requires RGB input)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process image with FaceMesh
with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            print("Facial Part Landmark Positions:")
            for part, indices in FACE_PARTS.items():
                print(f"\n{part}:")
                for i, index in enumerate(indices):
                    landmark = face_landmarks.landmark[index]
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    print(f"  Point {i+1}: ({x}, {y})")

                    # Draw circles on key points
                    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # Green dots for key points
                    cv2.putText(image, f"{i+1}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# Show the image with labeled key points
cv2.imshow("Face Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
