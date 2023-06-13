# 🔱 AI-Plays-God-of-War 💪🪓🏹🏺🐉⚔️

## Overview 😄📜 
Welcome, brave warrior, to the magical realm of AI-Plays-God-of-War! 🎮 🎉 This project utilizes the power of AI to play the popular game God of War. With the help of a LLM agent, an image captioning model, and YOLOv8 for object detection, our AI takes control of the main character, Kratos, and guides his actions in the game. You can try this out for free using a Cohere trial API Key. This README will provide you with step-by-step instructions on how to implement this AI in any game of your choice with minimal effort. Prepare yourself for an adventure filled with excitement and triumph! 😄🛡️🎮🤖

![demo](https://github.com/AkshitIreddy/AI-Plays-God-of-War/assets/90443032/e73fb8b3-3fa6-4fac-81a3-4c71b29db472)

Demo: https://youtu.be/yjzFgVBY0TE

## Prerequisites 💭📝
Before we dive into the exciting details, let's make sure we have everything set up:
#### 1. Cohere Trial API Key: To get started, sign up for a Cohere account and obtain your trial API key. This key will let you use Cohere's Language models for free.
#### 2. Game and Tools: Make sure you have God of War installed on your system. Additionally, we'll be using tools like image captioning models and YOLOv8, so install all the requirements from requirements.txt.
#### 3. GPT-4 Text and Image API (Optional): If you have access to the GPT-4 Text and Image API, you can enhance the AI capabilities by using the image input directly with GPT-4, replacing the image captioning model. This step is optional but can provide more accurate and seamless integration between the AI and the game.

## Project Structure 🚀🌐
Let's take a quick look at the structure of our project. It consists of several components and steps that we'll cover in detail:

#### [🍪 Creating a Custom YOLOv8 Model](#1)
#### [🍩 Preprocessing Training Videos](#2)
#### [🍰 Creating the Dataset](#3)
#### [🧁 Training the Object Detection Model](#4)
#### [🍭 Modifying the LLM Agent and Play Functions](#5)
Now, let's dive into each step and explore how each step works. 🤖🎮

## ✨ Step 1: Creating a Custom YOLOv8 Model<a name='1'></a>
### ✨ First Steps on the Path of the Gods ✨
To begin your epic quest, you must forge a YOLOv8 model capable of identifying characters, enemies, and other NPCs within the game. To enable the AI to detect characters and objects in the game, we'll start by creating a custom YOLOv8 model. This model will be trained to identify our main character, enemy characters, and other non-playable character (NPC) types based on your specific needs.

To create this model, we'll need videos of each character recorded individually. Ideally, these videos should cover various locations, different armor/skin combinations, and different poses if possible. If your game supports a photo mode, you can utilize that to capture the character from different angles.

For simplicity, instead of manually identifying bounding boxes for each character in each frame of the video, we'll leverage object detection techniques. We'll use an object detection model to detect characters in the videos and use the resulting bounding box coordinates as training data. Since most characters in games have a humaniod body they will be detected as humans and we can use those co-ordinates. This significantly reduces the manual effort involved.

## 🌌 Step 2: Preprocessing Training Videos<a name='2'></a>
### 🌌 Unveiling the Mystical Rituals of Video Preprocessing 🌌
Once we have recorded the videos for each character, we'll preprocess them to remove frames where no detections are found. This step helps us clean the data and ensures we only include frames where the model can accurately detect the character's position. Through the use of the magical "preprocess_training_video" function in the hallowed script known as "preprocess.py," we shall cleanse our videos of imperfections. You can also use this to figure out the correct threshold value where the model is able to detect the position of our character which will come in handy when you are creating the dataset. 
```sh
from functions.preprocess import preprocess_training_video

# Specify the path to your video file
video_path = "training_videos/atreus_training_video.mp4"

# Call the function to perform object detection on the video
preprocess_training_video(video_path)
```

## 📦 Step 3: Creating the Dataset<a name='3'></a>
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

## 🔥 Step 4: Training the Object Detection Model<a name='4'></a>
### 🔥 Igniting the Flames of Training 🔥
With the dataset in our possession, we now embark upon the holy act of training. Now comes the exciting part—training our object detection model! We'll use the powerful YOLOv8 model to train on the dataset we created in the previous step. This model will learn to identify our main character, enemy characters, and other NPCs based on the training data. The "train_model" function, revered within the tome "train.py," shall guide us through this challenging yet rewarding journey. Prepare the path to the dataset YAML file, set the number of training epochs, and determine the image size, for these choices shall shape the destiny of our model. Invoke the "train_model" function, passing these sacred artifacts as offerings. Witness the evolution of our object detection model, as it learns to perceive the true essence of our characters and adversaries.
```sh
from functions.train import train_model

# Set the path to the dataset YAML file
dataset_yaml = "dataset.yaml"

# Set the number of training epochs and image size
num_epochs = 100

# Set imgsz 
img_size = 640

# Train the model using the dataset
train_model(dataset_yaml, num_epochs, img_size)
```

## 🤖 Step 5: Modifying the LLM Agent and Play Functions<a name='5'></a>
### 🤖 Awakening the AI Agent 🤖
Behold, the time has come to awaken the AI agent that shall bring our characters to life! With our object detection model trained, it's time to integrate it into the game control mechanism. We'll modify the agent.py and play.py scripts to accommodate the new model and enable the AI to control the main character effectively. In agent.py, we'll add our Cohere trial API key to the llm_agent funtion. In play.py, based on the final object detection model's name and the keys responsible character's movement, special moves and attack, we'll modify the function. We'll use the llm_agent function to control the character's attacks and special moves. Depending on the detected positions of the main character and enemy characters, the LLM Agent will determine the appropriate actions for the character to take. 
(Note I used two object detection models in play_game function, one for Main character and one for enemies because my training data was small, about 2 mins for each character, but you can just use one model if you're dataset is large enough and you're characters are human-like which was wasn't the case for me as Draugr's are monsters.)

**Example of moving Player**
```sh
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
```

**Example of doing attacks or special moves**
```sh            
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
```

## ⚔️ Playing the Game with AI
### ⚔️ The Dance of Victory ⚔️
With our preparations complete, we stand poised to enter the game and unleash our newfound powers! The "play_game" function, an enchanting creation from the tome "play.py," shall guide our character through the trials and tribulations of the virtual world.

The mighty YOLOv8 model shall detect the positions of our character and enemies, while the ingenious "Key.py" script, birthed from the sacred teachings of sentdex's CyberPython Tutorial, shall guide our hero towards their foes. The LLM agent, a true embodiment of intelligence, shall determine the character's actions and maneuvers.

Behold the code that shall set our character in motion, navigating the virtual realm with grace and power:
```sh
from functions.play import play_game
import time

# sleep for some time to give me time to open the game
time.sleep(100)

# Call the function to perform object detection on the screen
play_game()
```

## ✨ Conclusion
### 🎉 A Tale of Legends, Laughter, and Triumph 🎉
Congratulations, noble warrior, for completing this mystical journey through the realm of AI-Plays-God-of-War! We have traversed the depths of artificial intelligence, weaving magic and technology into a tapestry of triumph. Together, we have harnessed the power of YOLOv8, trained a formidable object detection model, and awakened a LLM agent to guide our every move.

Now, armed with knowledge and wit, you are ready to embark upon your own adventures. Unleash the power of AI in your favorite games, conquer virtual realms, and make the gods themselves bow before your prowess.

Now, grab your controller, embark on an epic journey, and let the AI lead you to victory! May the gods be with you! 🌟⚔️🎮




