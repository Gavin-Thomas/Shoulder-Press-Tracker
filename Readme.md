# Shoulder Press & Computer Vision

The following is my attempt at creating a small python application that uses OpenCV, MediaPipe, NumPy, and Python to make an app that will track my
movement and count reps of my shoulder press. Hypothetically this app could be used to find squat depth (angle) and count squats, or any other type of
tracking as well.

I use a lot of commented code during this walkthrough, so please bear with me. Most explanations are given with my commented code.

## Install and Import Necessary Libraries

### First I installed mediapipe, which is a general "detection" pre-built ML library

```
pip install mediapipe
```

### Next I installed opencv, which is a 'recognition' library for different types of computer vision

```
pip install opencv-python
```

```
# Now lets import OpenCV and Mediapipe
import cv2
import mediapipe as mp
# Numpy for Trig :)
import numpy as np

# A mediapipe solution is like a model that we are grabbing
mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

### A disclaimer: I continue to build on this code throughout so you can see my thought process
```

### Now for the camera setup and getting the colours adjusted... 
```
# Here is our VIDEO FEED

# Setup video capture device
cap = cv2.VideoCapture(0)

# This loops through our video feed 
while cap.isOpened():
    # cap.read is like us getting the current feed from webcam
    # frame variable gives us the image from our webcam, ret is return var
    ret,frame = cap.read()
    #pipe through mediapipe feed, and image from webcam (frame)
    cv2.imshow('Mediapipe Feed',frame)

    # check if we try to break out of screen, this function helps us break out of the while loop
    # 0xFF helps us figure out which key we are hitting on our keyboard
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# if we break out of our video feed we will release our video feed and destroy windows
cap.release()
cv2.destroyAllWindows()
```
- What does this look like?
![NormalVideoFeed](https://github.com/Gavin-Thomas/KNES-381/blob/main/images/Normal%20Feed.png?raw=true)

## Now Let's Make some Body Outline Detections

### I added this block of code to the video feed...

```
        # Let's Render our image detection now
        # No need to draw point, by point, just use media pipe drawing utilities
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                # Dot colour and specifications
                                mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                                # Line colour and specifications
                                mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))
```
- Now we add that block to our video feed code window

```
# Here is our VIDEO FEED

# Setup video capture device
cap = cv2.VideoCapture(0)

# Setup our detection confidence and our tracking confidence
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    # Loop through our video feed 
    while cap.isOpened():
        # cap.read is like us getting the current feed from webcam
        # frame variable gives us the image from our webcam, ret is return var
        ret,frame = cap.read()

        # Let's try and make some detections now...
        # First we recolour our image
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # This line makes the detection (we set up our pose model before)
        # We store detections inside results
        results = pose.process(image)

        # Write up image then colour the image again back to normal image format
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Let's Render our image detection now
        # No need to draw point, by point, just use media pipe drawing utilities
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                # Dot colour and specifications
                                mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                                # Line colour and specifications
                                mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))


        #pipe through mediapipe feed, and image from webcam (frame)
        cv2.imshow('Mediapipe Feed',image)

        # check if we try to break out of screen, this function helps us break out of the while loop
        # 0xFF helps us figure out which key we are hitting on our keyboard
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# if we break out of our video feed we will release our video feed and destroy windows
cap.release()
cv2.destroyAllWindows()
```
- What does the detection of mediapipe look like?
![Detection](https://github.com/Gavin-Thomas/KNES-381/blob/main/images/Detection_Feed.png?raw=true)


## Next up, let's figure out our JOINTS

```
        # This block of code allows for the extraction of landmarks
        # if we don't have detections or get an error we just pass through
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass
```
- We'll now render our image detection

```
        # Let's Render our image detection now
        # No need to draw point, by point, just use media pipe drawing utilities
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                # Dot colour and specifications
                                mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                                # Line colour and specifications
                                mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))
```
- Let's add this code block to our video feed now...

```
# Here is our VIDEO FEED

# Setup video capture device
cap = cv2.VideoCapture(0)

# Setup our detection confidence and our tracking confidence
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    # Loop through our video feed 
    while cap.isOpened():
        # cap.read is like us getting the current feed from webcam
        # frame variable gives us the image from our webcam, ret is return var
        ret,frame = cap.read()

        # Let's try and make some detections now...
        # First we recolour our image
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # This line makes the detection (we set up our pose model before)
        # We store detections inside results
        results = pose.process(image)

        # Write up image then colour the image again back to normal image format
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # This block of code allows for the extraction of landmarks
        # if we don't have detections or get an error we just pass through
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass
        
        # Let's Render our image detection now
        # No need to draw point, by point, just use media pipe drawing utilities
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                # Dot colour and specifications
                                mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                                # Line colour and specifications
                                mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))


        #pipe through mediapipe feed, and image from webcam (frame)
        cv2.imshow('Mediapipe Feed',image)

        # check if we try to break out of screen, this function helps us break out of the while loop
        # 0xFF helps us figure out which key we are hitting on our keyboard
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# if we break out of our video feed we will release our video feed and destroy windows
cap.release()
cv2.destroyAllWindows()
```
### Great, now we have landmarks setup. 

```
# How many landmarks??
len(landmarks)
```

```
#print each landmark from particular landmark map
for lndmrk in mp_pose.PoseLandmark:
    print(lndmrk)
    
# This is more or less, example code to show how I determined the names of the different landmarks (ie. LEFT_HIP, or LEFT_SHOULDER)
```

- As an example, let's get our shoulder values

```
# Access the left shoulder landmark as loop and display it from array
landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

# OUTPUT
x: 0.7799567
y: 0.84146
z: -0.41408777
visibility: 0.99849945

# I will just use the x and y coordinates for this project. Hypothetically you could
# (continued) do a bunch of other cool stuff with the z coordinate and the joint visibility!
```

### Now for the Fun Part! Calculating Angles...
- The following block of code is some trigonometry. First we define a function to get take the first joint (a), middle joint (b), and last joint (c). Then we find the angle between them by getting the radians and then the absolute angle!

```
# Let's define a function to calculate the angle between the left hip, left shoulder, and left elbow
def calculate_angle(a,b,c):
    a = np.array(a) # First Joint
    b = np.array(b) # Middle Joint
    c = np.array(c) # End Joint

    # Now lets do some Trigonometry!
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    # Our hinge joints cannot go more than 360 degrees hence this equation
    if angle >180.0:
        angle = 360-angle
        
    return angle 
```

- This next block of code, we will add to our video feed. This next block of code is us grabbing our x and y values from the left hip, shoulder, and elbow. We want to find the angle between these joints to calculate the shoulder press angle.

```
# Get Left Shoulder x and y values
hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
```

- Quick test of the function
```
calculate_angle(hip, shoulder, elbow)

#OUTPUT ANGLE - just to make sure function works
15.541339769588362
```
- This next step is important for finding my webcam feed dimensions
```
# Let's just quickly find out what my width and height of video feed is!
# Initialize video capture device
cap = cv2.VideoCapture(0)

# Get width and height of video feed
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Print width and height of video feed
print(f"Video feed width: {width}, height: {height}")

# Release video capture device
cap.release()

```

### However, the ANGLES are not yet rendered into our video feed, so we need to add this to the feed. - we need to add the "calculate_angle" function, and we need to add all of our landmark coordinates.

```
# Here is our VIDEO FEED

# Setup video capture device
cap = cv2.VideoCapture(0)

# Setup our detection confidence and our tracking confidence
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    # Loop through our video feed 
    while cap.isOpened():
        # cap.read is like us getting the current feed from webcam
        # frame variable gives us the image from our webcam, ret is return var
        ret, frame = cap.read()

        # Let's try and make some detections now...
        # First we recolour our image
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # This line makes the detection (we set up our pose model before)
        # We store detections inside results
        results = pose.process(image)

        # Write up image then colour the image again back to normal image format
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # This block of code allows for the extraction of landmarks
        # if we don't have detections or get an error we just pass through

        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get the X and Y values of the hip, shoulder, and elbow
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            
            # Calculate angle
            angle = calculate_angle(hip, shoulder, elbow)
            
            # Now let's visualize our angle data (pass through image from webcam and angle)
            # Use array multiplication (grab elbow coordinate and multiply by my webcam dimensions)
            # We do the multiplication to get the proper image dimensions (then we convert to a tuple which cv2 expects)
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(shoulder, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                       
        except:
            pass
        
        # Let's Render our image detection now
        # No need to draw point, by point, just use media pipe drawing utilities
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                # Dot colour and specifications
                                mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                                # Line colour and specifications
                                mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))


        #pipe through mediapipe feed, and image from webcam (frame)
        cv2.imshow('Mediapipe Feed',image)

        # check if we try to break out of screen, this function helps us break out of the while loop
        # 0xFF helps us figure out which key we are hitting on our keyboard
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# if we break out of our video feed we will release our video feed and destroy windows
cap.release()
cv2.destroyAllWindows()
```

- Let's take a look at our joint angle. The joint angle shown (and what will be used for the shoulder press tracker) is the angle between the hip, shoulder, and elbow. I haven't yet rounded the angle.
![joint_angle](https://github.com/Gavin-Thomas/KNES-381/blob/main/images/Joint_Angle_feed.png?raw=true)

### FINAL STEP: The shoulder press tracker

First code block, set the counter to 0, and stage to None. We put this block of code at the top of the video feed.
```
# Count Shoulder Press Reps
counter = 0
# Stage represents the up or down phase of the curl
stage = None
```
- We put this logic block of code after the "angle" is defined

```
# Now for some logical if statements for the shoulder press counter
            if angle > 160:
                stage = 'up'
            if angle < 80 and stage == 'up':
                stage = 'down'
                counter += 1
                print(counter)
                       
        except:
            pass
 ```
 
 - Then we add in the GUI for the counter.

```
        # Now let's render the shoulder press ticker
        # To do this let's first setup a status box (last line ='-1' fills box with colour)
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Now for our REP data
        cv2.putText(image,'Rep',(15,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 2, cv2.LINE_AA)            
        
        # Now for our STAGE data
        cv2.putText(image,'Stage',(65,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 2, cv2.LINE_AA) 

```

- Once we add in this block to our video feed code block, we should be good to go!

```

# Count Shoulder Press Reps
counter = 0
# Stage represents the up or down phase of the curl
stage = None



# Here is our VIDEO FEED
# Setup video capture device
cap = cv2.VideoCapture(0)

# Setup our detection confidence and our tracking confidence
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    # Loop through our video feed 
    while cap.isOpened():
        # cap.read is like us getting the current feed from webcam
        # frame variable gives us the image from our webcam, ret is return var
        ret, frame = cap.read()

        # Let's try and make some detections now...
        # First we recolour our image
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # This line makes the detection (we set up our pose model before)
        # We store detections inside results
        results = pose.process(image)

        # Write up image then colour the image again back to normal image format
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # This block of code allows for the extraction of landmarks
        # if we don't have detections or get an error we just pass through

        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get the X and Y values of the hip, shoulder, and elbow
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            
            # Calculate angle
            angle = calculate_angle(hip, shoulder, elbow).round(3)
            
            # Now let's visualize our angle data (pass through image from webcam and angle)
            # Use array multiplication (grab elbow coordinate and multiply by my webcam dimensions)
            # We do the multiplication to get the proper image dimensions (then we convert to a tuple which cv2 expects)
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(shoulder, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            # Now for some logical if statements for the shoulder press counter
            if angle > 160:
                stage = 'up'
            if angle < 80 and stage == 'up':
                stage = 'down'
                counter += 1
                print(counter)
                       
        except:
            pass
        
        # Now let's render the shoulder press ticker
        # To do this let's first setup a status box (last line ='-1' fills box with colour)
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Now for our REP data
        cv2.putText(image,'Rep',(15,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 2, cv2.LINE_AA)            
        
        # Now for our STAGE data
        cv2.putText(image,'Stage',(65,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 2, cv2.LINE_AA) 

        # Let's Render our image detection now
        # No need to draw point, by point, just use media pipe drawing utilities
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                # Dot colour and specifications
                                mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                                # Line colour and specifications
                                mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))


        #pipe through mediapipe feed, and image from webcam (frame)
        cv2.imshow('Mediapipe Feed',image)

        # check if we try to break out of screen, this function helps us break out of the while loop
        # 0xFF helps us figure out which key we are hitting on our keyboard
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# if we break out of our video feed we will release our video feed and destroy windows
cap.release()
cv2.destroyAllWindows()
```
- Okay, let's take a look and see if our tracker is there on the top left of the screen...
![tracker](https://github.com/Gavin-Thomas/KNES-381/blob/main/images/Gui_Setup.png?raw=true)

# Watch the Rep Counter in Action! (image links to my youtube)

[![GotoYoutube](https://github.com/Gavin-Thomas/KNES-381/blob/main/images/Youtubelink.png?raw=true)](https://www.youtube.com/watch?v=e-XoiDOFD5s "Video Title")


