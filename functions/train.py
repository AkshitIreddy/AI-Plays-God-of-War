from ultralytics import YOLO

def train_model(dataset_yaml, num_epochs, img_size):
    # Load the YOLO model
    model = YOLO()

    # Train the model
    model.train(data=dataset_yaml, epochs=num_epochs, imgsz=img_size)