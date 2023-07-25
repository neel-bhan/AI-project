import pygame
import cv2
import torch
from PIL import Image
from torchvision import transforms
from network_file import Net
import numpy as np
import copy
PATH = r'sign_lang_1.pth'
classes =['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
global index
index = 12
def display_text():
    ret, frame_print = cap.read()

    frame_print = Image.fromarray(cv2.cvtColor(frame_print, cv2.COLOR_BGR2RGB))

    # Apply transformations
    image = transform(frame_print)

    # Add an extra dimension because the Net expects batches
    image = image.unsqueeze(0)

    # Make a prediction
    output = net(image)

    # Get the predicted class
    _, predicted_class = torch.max(output, 1)
    letter = classes[predicted_class.item()]
    print(letter)
    font = pygame.font.Font(None, 74)
    text = font.render(letter, 1, (0, 0, 0))
    writing_page.blit(text, (20+index, 20))


    pygame.display.update()
    pygame.display.flip()


net = Net()

net.load_state_dict(torch.load(PATH))
net.eval()
pygame.init()

# Define screen dimensions and create Pygame screen
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

webcam_screen = pygame.Surface((370 , 270 ))
prediction_screen = pygame.Surface((370 , 270 ))
writing_page = pygame.Surface((760, 270 ))
writing_page.fill((102, 204, 255))
# Initialize webcam
cap = cv2.VideoCapture(0)

classes =['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# Define the margin
margin = 20
imp = pygame.image.load("back.jpg").convert()

# Using blit to copy content from one surface to other
screen.blit(imp, (0, 0))
transform = transforms.Compose(
    [transforms.Resize((200, 200)), # Resize the images to 224x224 pixels
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.5190, 0.4977, 0.5134), std=(0.2028, 0.2328, 0.2416))])


# Main loop
running = True
while running:
    # Event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_k:
                display_text()
                index += 35


    # Capture frame-by-frame
    ret, frame = cap.read()
    pred_frame = copy.deepcopy(frame)


    if ret:
        # Our operations on the frame come here
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, ( 370, 270))
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Convert image from OpenCV BGR format to Pygame RGB format
        frame = pygame.surfarray.make_surface(frame)


        # Display the resulting frame
        webcam_screen.blit(frame, (0,0))

    pred_frame = Image.fromarray(cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB))

    # Apply transformations
    image = transform(pred_frame)

    # Add an extra dimension because the Net expects batches
    image = image.unsqueeze(0)

    # Make a prediction
    output = net(image)

    # Get the predicted class
    _, predicted_class = torch.max(output, 1)

    prediction_screen.fill((102, 204, 255))


    # Clear the window


    # Draw the predicted letter
    font = pygame.font.Font(None, 74)
    text = font.render(classes[predicted_class.item()], 1, (255, 255, 255))
    prediction_screen.blit(text, (165, 115))

    # Update the window
    screen.blit(webcam_screen, (20, 20))
    screen.blit(prediction_screen, (410, 20))
    screen.blit(writing_page, (20, 310))
    pygame.display.update()
    pygame.display.flip()


# When everything done, release the capture and quit Pygame
cap.release()
pygame.quit()
