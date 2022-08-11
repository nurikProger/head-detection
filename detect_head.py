import cv2
import mediapipe as mp
import math
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# gets the coordinates of 6 different spots on face
def get_points(coordinates):
    items = coordinates.split('\n')

    points = []
    xy = []

    for i in items:

        if "x:" in i:
            xy = []
            x = float(i.split()[1])
            xy.append(x)

        elif "y:" in i:
            y = float(i.split()[1])
            xy.append(y)
            points.append(xy)

    return points




# For webcam input:
cap = cv2.VideoCapture(0) # Initializing the camera

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # dimensions of the frame
        window_x = image.shape[1]
        window_y = image.shape[0]

        # making the frame unwriteable to improve the performance
        image.flags.writeable = False

        # procoss the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # making the frame writeable again to draw on it
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        if results.detections:
            for detection in results.detections:

                # sequence of values: Right Eye, Left Eye, Nose, Mouth, Right End, Left End
                points = get_points(str(detection))

                # Box around the face
                # the coordinates of Nose
                nose_x = int(points[2][0] * window_x)
                nose_y = int(points[2][1] * window_y)

                # the coordinates of Right End
                rightEnd_x = int(points[4][0] * window_x)
                rightEnd_y = int(points[4][1] * window_y)

                # the coordinates of Left End
                leftEnd_x = int(points[5][0] * window_x)
                leftEnd_y = int(points[5][1] * window_y)

                # distance between the ends
                distance = int(math.sqrt((rightEnd_x - leftEnd_x)**2 + (rightEnd_y - leftEnd_y)**2))

                # box
                box_start = (rightEnd_x, nose_y - int(distance/2))
                box_end = (rightEnd_x + distance, rightEnd_y + int(distance/2))

                # draw the box
                cv2.rectangle(image, box_start, box_end, (0,255,0), 6)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Face Detection', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()