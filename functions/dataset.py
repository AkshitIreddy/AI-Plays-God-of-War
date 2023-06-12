import os
import cv2
import yaml
from ultralytics import YOLO

def create_dataset(video_list, threshold_list):
    # Create the directory structure
    base_dir = "datasets"
    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "valid")
    train_images_dir = os.path.join(train_dir, "images")
    train_labels_dir = os.path.join(train_dir, "labels")
    valid_images_dir = os.path.join(valid_dir, "images")
    valid_labels_dir = os.path.join(valid_dir, "labels")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)

    # Load the YOLO model
    model = YOLO()

    for video_index, video_path in enumerate(video_list):
        
        threshold = threshold_list[video_index]

        # Extract the video name as the character name
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        character_name = video_name

        # Open the video file
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Process each frame
        for frame_index in range(frame_count):
            success, frame = video.read()
            if not success:
                break

            # Perform object detection
            results = model(frame)

            # Extract bounding boxes and confidences
            boxes = results[0].boxes
            confidences = boxes.conf

            # Filter detections with confidence above threshold
            filtered_indices = confidences > threshold
            filtered_boxes = boxes[filtered_indices]

            # Save the image and label file
            image_filename = f"{video_index}_{frame_index}.jpg"
            label_filename = f"{video_index}_{frame_index}.txt"

            if frame_index % 10 == 0:
                # Validation image
                image_path = os.path.join(valid_images_dir, image_filename)
                label_path = os.path.join(valid_labels_dir, label_filename)
            else:
                # Training image
                image_path = os.path.join(train_images_dir, image_filename)
                label_path = os.path.join(train_labels_dir, label_filename)

            cv2.imwrite(image_path, frame)

            with open(label_path, "w") as label_file:
                for box in filtered_boxes:
                    x1, y1, x2, y2 = box.xyxy.round().int().tolist()[0][:4]
                    # Save the object coordinates and class as the video name
                    label_file.write(f"{video_index} {((x1 + x2) / 2 / frame.shape[1]):.6f} {((y1 + y2) / 2 / frame.shape[0]):.6f} {(x2 - x1) / frame.shape[1]:.6f} {(y2 - y1) / frame.shape[0]:.6f}\n")

    # Generate the YAML file for the dataset
    yaml_path = "dataset.yaml"
    num_classes = len(video_list)
    class_names = [os.path.splitext(os.path.basename(video))[0] for video in video_list]

    dataset_config = {
        "train": 'train',
        "val": 'valid',
        "nc": num_classes,
        "names": class_names
    }

    with open(yaml_path, "w") as yaml_file:
        yaml.dump(dataset_config, yaml_file, default_flow_style=False)