import random
import cv2
from time import time, sleep
import functions.keys as k
from ultralytics import YOLO
import pyautogui
import numpy as np
from functions.agent import llm_agent

def capture_screen():
    screen = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
    return frame

def play_game():
    keys = k.Keys()

    # Load the YOLO models
    kratos_model = YOLO("Kratos.pt")
    other_objects_model = YOLO("enemy.pt")

    # Create VideoWriter object to save the output video
    frame_width, frame_height = 1920, 1080
    output_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while True:
        # Capture the screen
        screen = capture_screen()

        # Get screen dimensions
        if frame_width is None or frame_height is None:
            frame_height, frame_width, _ = screen.shape

        # Perform object detection on the screen using the Kratos model
        kratos_results = kratos_model(screen)

        # Extract bounding boxes and confidences for Kratos detections
        kratos_boxes = kratos_results[0].boxes
        kratos_confidences = kratos_boxes.conf

        # Check if there are any Kratos detections
        if len(kratos_confidences) != 0:
            # Find the index of the detection with the highest confidence
            kratos_index = kratos_confidences.argmax()
            # Get the coordinates of the Kratos bounding box
            kratos_x1, kratos_y1, kratos_x2, kratos_y2 = kratos_boxes.xyxy.round().int().tolist()[kratos_index][:4]
            # Draw bounding box and label for Kratos
            cv2.rectangle(screen, (kratos_x1, kratos_y1), (kratos_x2, kratos_y2), (0, 0, 255), 2)
            kratos_label = f"Class: Kratos LLM Agent Control Mode (Confidence: {kratos_confidences[kratos_index]:.2f})"
            cv2.putText(screen, kratos_label, (kratos_x1, kratos_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            kratos_x1, kratos_y1, kratos_x2, kratos_y2 = 0, 0, 1920, 1080

        # Perform object detection on the screen using the other objects model
        other_objects_results = other_objects_model(screen)

        # Extract bounding boxes and confidences for other objects detections
        other_objects_boxes = other_objects_results[0].boxes
        other_objects_confidences = other_objects_boxes.conf

        # Display the image with bounding boxes and labels
        for box in other_objects_boxes:
            x1, y1, x2, y2 = box.xyxy.round().int().tolist()[0][:4]
            confidence = box.conf.item()
            class_id = box.cls.item()

            # Draw bounding box rectangle
            cv2.rectangle(screen, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"Class: {other_objects_results[0].names[int(class_id)]}"
            cv2.putText(screen, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Move Kratos towards the nearest person
        if len(other_objects_boxes) > 0:
            kratos_center_x = int((kratos_x1 + kratos_x2) / 2)
            kratos_center_y = int((kratos_y1 + kratos_y2) / 2)

            nearest_person = other_objects_boxes[0]
            person_x1, person_y1, person_x2, person_y2 = nearest_person.xyxy.round().int().tolist()[0][:4]
            person_center_x = int((person_x1 + person_x2) / 2)
            person_center_y = int((person_y1 + person_y2) / 2)

            # Calculate the distance between Kratos and the nearest person
            distance_x = person_center_x - kratos_center_x
            distance_y = person_center_y - kratos_center_y

            # Check if a person is very close to Kratos
            if abs(distance_x) < 50 or abs(distance_y) < 50:
                # Select an action
                action = llm_agent(screen) 
                if action == "light attack":
                    # Left mouse click (attack)
                    keys.directMouse(buttons=keys.mouse_lb_press)
                    sleep(0.5)
                    keys.directMouse(buttons=keys.mouse_lb_release)
                elif action == "heavy attack":
                    # Right mouse click (attack)
                    keys.directMouse(buttons=keys.mouse_rb_press)
                    sleep(0.5)
                    keys.directMouse(buttons=keys.mouse_rb_release)
                elif action == "dodge back":
                    # Move in the opposite direction to the NPC
                    direction = random.choice(["w", "a", "s", "d"])
                    keys.directKey(direction)
                    sleep(0.04)
                    keys.directKey(direction, keys.key_release)
                    # Press the space bar twice
                    keys.directKey('0x39')
                    sleep(0.04)
                    keys.directKey("0x39", keys.key_release)
                    sleep(0.04)
                    keys.directKey('0x39')
                    sleep(0.04)
                    keys.directKey("0x39", keys.key_release)

            # Move Kratos towards the nearest person
            if distance_x < 0:
                keys.directKey("a")
                sleep(1)
                keys.directKey("a", keys.key_release)
            elif distance_x > 0:
                keys.directKey("d")
                sleep(1)
                keys.directKey("d", keys.key_release)

            if distance_y < 0:
                keys.directKey("w")
                sleep(1)
                keys.directKey("w", keys.key_release)
            elif distance_y > 0:
                keys.directKey("s")
                sleep(1)
                keys.directKey("s", keys.key_release)
        else:
            # Move the camera if no person is detected
            keys.directMouse(-20, 0)
            sleep(0.04)
            
        # Write the frame with bounding boxes to the output video
        output_video.write(screen)

        # Display the frame with bounding boxes and labels
        cv2.imshow('Object Detection', screen)

        # Wait for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture, output video, and close the window
    output_video.release()
    cv2.destroyAllWindows()
