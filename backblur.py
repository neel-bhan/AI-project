import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from CNN_CLASS import Net
import numpy as np
import time

# Define the transformations - these should be the same as what you used during training
transform = transforms.Compose([
    transforms.Resize((200, 200)),  # Resize to 400x400
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize the model
model = Net(26)

# Load the saved parameters
model.load_state_dict(torch.load('finalcnn.pth'))  # Set your correct path here

# Set the model to evaluation mode
model.eval()

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # 0 is the index of the webcam. Adjust if necessary.

# Define the list of class names
classes =['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# Loop for processing the video frames
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to PIL for easier transformations
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply transformations
    image = transform(frame)

    # Add an extra dimension because the model expects batches
    image = image.unsqueeze(0)

    # Make a prediction
    output = model(image)

    # Get the predicted class
    _, predicted_class = torch.max(output, 1)
    print(f'Predicted class: {predicted_class.item()}')
    print(classes[predicted_class.item()])

    # Display the resulting frame
    cv2.imshow('frame', cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


import pixellib
from pixellib.tune_bg import alter_bg


capture = cv2.VideoCapture(0)
change_bg = alter_bg(model_type = "pb")
change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
change_bg.gray_camera(capture, frames_per_second=10, show_frames = True, frame_name = "frame",
output_video_name="output_video.mp4", detect = "person")
# Release the webcam and destroy all windows once done
cap.release()
cv2.destroyAllWindows()
