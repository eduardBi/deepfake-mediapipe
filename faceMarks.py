import cv2
import mediapipe as mp
import sys

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define facial landmarks based on user selection
FACE_PARTS = {
    "Nose": [1, 8, 236, 456],
    "Chin": [148, 152, 377],
    "Lashes": [46, 55, 52, 285, 282, 276],
    "Left Eye":[113,27,190,23],
    "Right Eye":[414,253,257,342],
    "left lash":[46,52,55],
    "right lash":[285,282,276],
    "lips left corner":[61],
    "lips right corner":[291],
    "forehead":[109, 10, 338],
    "left cheek":[116],
    "right cheek":[345],
    "top lip top part":[37,0,267],
    "bottom lip top part":[87,14,317],
    "bottom lip bottom part":[84,17,314],
    "top lip bottom part":[82,12,312],
    "left grinners":[216,202,212],
    "right  grinners":[436,432,422],

}

try:
    windows = []  # Store window names

    for i in range(333, 349):  # Loop through images 333.png to 344.png
        image_path = f"cropped_faces/face_{i}_0.png"
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not read image at '{image_path}'")
            continue  # Skip missing images

        # Convert to RGB (MediaPipe requires RGB input)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image with FaceMesh
        with mp_face_mesh.FaceMesh(
                static_image_mode=True, 
                max_num_faces=1, 
                refine_landmarks=True,  
                min_detection_confidence=0.7  
            ) as face_mesh:
            results = face_mesh.process(rgb_image)

            # Check if any face landmarks were detected
            if not results.multi_face_landmarks:
                print(f"No marks were found in '{image_path}'")
                continue  # Skip to next image



            # If landmarks are found, process and display them
            for face_landmarks in results.multi_face_landmarks:
                print(f"\nFacial Part Landmark Positions for {image_path}:")
                for part, indices in FACE_PARTS.items():
                    print(f"\n{part}:")
                    for idx, index in enumerate(indices):
                        landmark = face_landmarks.landmark[index]
                        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                        print(f"  Point {index}: ({x}, {y})")

                        if part =='left grinners':
                            
                        elif part =='right grinners':


                        # Draw circles on key points
                        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # Green dots for key points

        # Store the window name
        window_name = f"Face Landmarks - {i}"
        windows.append(window_name)
        cv2.imshow(window_name, image)  # Show image in a separate window

    # Keep all windows open until 'q' is pressed
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit loop when 'q' is pressed

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("\nExit: Ctrl + C detected. Exiting gracefully.")
    sys.exit()
