# 🔱 AI-Plays-God-of-War 💪🪓🏹🏺🐉⚔️

## Overview 😄📜 
Welcome, brave warrior, to the magical realm of AI-Plays-God-of-War! 🎮 🎉 This project utilizes the power of AI to play the popular game God of War. With the help of a LLM agent, an image captioning model, and YOLOv8 for object detection, our AI takes control of the main character, Kratos, and guides his actions in the game. You can try this out for free using the Cohere trial API Key. This README will provide you with step-by-step instructions on how to implement this AI in any game of your choice with minimal effort. Prepare yourself for an adventure filled with excitement and triumph! 😄🛡️🎮🤖

## Prerequisites
Before we dive into the exciting details, let's make sure we have everything set up:
1. Cohere Trial API Key: To get started, sign up for a Cohere trial and obtain your trial API key. This key will let you use Cohere's Language models for free.
2. Game and Tools: Make sure you have God of War installed on your system. Additionally, we'll be using tools like image captioning models and YOLOv8, so install all the requirements from requirements.txt.
3. GPT-4 Text and Image API (Optional): If you have access to the GPT-4 Text and Image API, you can enhance the AI capabilities by using the image input directly with GPT-4, replacing the image captioning model. This step is optional but can provide more accurate and seamless integration between the AI and the game.

## Project Structure
Let's take a quick look at the structure of our project. It consists of several components and steps that we'll cover in detail:

#### 1. Creating a Custom YOLOv8 Model
#### 2. Preprocessing Training Videos
#### 3. Creating the Dataset
#### 4. Training the Object Detection Model
#### 5. Modifying the LLM Agent and Play Functions
Now, let's dive into each step and explore how each step works. 🤖🎮

## Step 1: Creating a Custom YOLOv8 Model
### ✨ First Steps on the Path of the Gods ✨
To begin your epic quest, you must forge a YOLOv8 model capable of identifying characters, enemies, and other NPCs within the game. To enable the AI to detect characters and objects in the game, we'll start by creating a custom YOLOv8 model. This model will be trained to identify our main character, enemy characters, and other non-playable character (NPC) types based on your specific needs.

To create this model, we'll need videos of each character recorded individually. Ideally, these videos should cover various locations, different armor/skin combinations, and different poses if possible. If your game supports a photo mode, you can utilize that to capture the character from different angles.

For simplicity, instead of manually identifying bounding boxes for each character in each frame of the video, we'll leverage object detection techniques. We'll use an object detection model to detect characters in the videos and use the resulting bounding box coordinates as training data. Since most characters in games have a humaniod body they will be detected as humans and we can use those co-ordinates. This significantly reduces the manual effort involved.

## Preprocessing Training Videos
### 🌌 Unveiling the Mystical Rituals of Video Preprocessing 🌌
Once we have recorded the videos for each character, we'll preprocess them to remove frames where no detections are found. This step helps us clean the data and ensures we only include frames where the model can accurately detect the character's position. Through the use of the magical "preprocess_training_video" function in the hallowed script known as "preprocess.py," we shall cleanse our videos of imperfections. You can also use this to figure out the correct threshold value where the model is able to detect the position of our character which will come in handy when you are creating the dataset. 
```sh
from functions.preprocess import preprocess_training_video

# Specify the path to your video file
video_path = "training_videos/atreus_training_video.mp4"

# Call the function to perform object detection on the video
preprocess_training_video(video_path)
```

## Step 3: Creating the Dataset
### 📦 Forging the Dataset 📦
Now that our videos have been purified, we can venture forth to create a magnificent dataset. With the video paths and threshold values in hand, we shall employ the mighty "create_dataset" function from the sacred tome known as "dataset.py" to generate a dataset that will serve as training data for our object detection model. Depending on which threshold performed best at identifying the character during the preprocessing step, we'll compile a list of video paths and corresponding threshold values. The generated dataset will contain the necessary information for training our custom object detection model.
```sh
from functions.dataset import create_dataset

# Set the paths to video_paths and set the values in threshold_list
video_paths = ['preprocessed_videos/Draugr.mp4', 'preprocessed_videos/Atreus.mp4', 'preprocessed_videos/Kratos.mp4']
threshold_list = [0.6, 0.7, 0.9]

# Create the dataset
create_dataset(video_paths, threshold_list)
```










































