import cv2
from ultralytics import YOLO

def preprocess_training_video(video_path):
    # Load the YOLO model
    model = YOLO("yolov8x.pt")

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Create VideoWriter object to save the output video
    output_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        # Read a frame from the video
        ret, frame = video.read()

        # If the frame was not read successfully, end the loop
        if not ret:
            break

        # Perform object detection
        results = model(frame)

        # make a copy
        clean_frame = frame

        # Extract bounding boxes and confidences
        boxes = results[0].boxes
        confidences = boxes.conf

        # Check if there are any detections
        if len(confidences) == 0:
            continue

        # Filter detections with confidence above the threshold
        threshold = 0.6
        filtered_indices = confidences > threshold
        filtered_boxes = boxes[filtered_indices]

        # Display the image with bounding boxes and labels
        for box in filtered_boxes:
            x1, y1, x2, y2 = box.xyxy.round().int().tolist()[0][:4]
            confidence = box.conf.item()
            class_id = box.cls.item()

            # Draw bounding box rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display label and confidence
            label = f"Class: {results[0].names[int(class_id)]}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame with bounding boxes and labels
        cv2.imshow('Object Detection', frame)

        # Write the frame with bounding boxes to the output video
        output_video.write(clean_frame)
        
        # Wait for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture, save the output video and close the window
    video.release()
    output_video.release()
    cv2.destroyAllWindows()