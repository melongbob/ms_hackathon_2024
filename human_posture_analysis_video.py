import cv2
import time
import math as m
import mediapipe as mp


# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


def getLandmarks(lm, w, h):
    lm = keypoints.pose_landmarks
    lmPose = mp_pose.PoseLandmark

    landmark = {
        "LEFT": {
            "EAR": {
                "x": int(lm.landmark[lmPose.LEFT_EAR].x * w), 
                "y": int(lm.landmark[lmPose.LEFT_EAR].y * h)
            },
            "SHOULDER": {
                "x": int(lm.landmark[lmPose.LEFT_SHOULDER].x * w), 
                "y": int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            },
            "ELBOW": {
                "x": int(lm.landmark[lmPose.LEFT_ELBOW].x * w), 
                "y": int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
            },
            "WRIST": {
                "x": int(lm.landmark[lmPose.LEFT_ELBOW].x * w), 
                "y": int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
            },
            "ELBOW": {
                "x": int(lm.landmark[lmPose.LEFT_ELBOW].x * w), 
                "y": int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
            },
            "WRIST": {
                "x": int(lm.landmark[lmPose.LEFT_WRIST].x * w), 
                "y": int(lm.landmark[lmPose.LEFT_WRIST].y * h)
            },
            "INDEX": {
                "x": int(lm.landmark[lmPose.LEFT_INDEX].x * w), 
                "y": int(lm.landmark[lmPose.LEFT_INDEX].y * h)
            },
            "HIP": {
                "x": int(lm.landmark[lmPose.LEFT_HIP].x * w), 
                "y": int(lm.landmark[lmPose.LEFT_HIP].y * h)
            },
            "KNEE": {
                "x": int(lm.landmark[lmPose.LEFT_KNEE].x * w), 
                "y": int(lm.landmark[lmPose.LEFT_KNEE].y * h)
            },
            "HEEL": {
                "x": int(lm.landmark[lmPose.LEFT_HEEL].x * w), 
                "y": int(lm.landmark[lmPose.LEFT_HEEL].y * h)
            },
            "FOOT_INDEX": {
                "x": int(lm.landmark[lmPose.LEFT_FOOT_INDEX].x * w), 
                "y": int(lm.landmark[lmPose.LEFT_FOOT_INDEX].y * h)
            }
        },
        "RIGHT": {
            "EAR": {
                "x": int(lm.landmark[lmPose.RIGHT_EAR].x * w), 
                "y": int(lm.landmark[lmPose.RIGHT_EAR].y * h)
            },
            "SHOULDER": {
                "x": int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w), 
                "y": int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            },
            "ELBOW": {
                "x": int(lm.landmark[lmPose.RIGHT_ELBOW].x * w), 
                "y": int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)
            },
            "WRIST": {
                "x": int(lm.landmark[lmPose.RIGHT_ELBOW].x * w), 
                "y": int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)
            },
            "ELBOW": {
                "x": int(lm.landmark[lmPose.RIGHT_ELBOW].x * w), 
                "y": int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)
            },
            "WRIST": {
                "x": int(lm.landmark[lmPose.RIGHT_WRIST].x * w), 
                "y": int(lm.landmark[lmPose.RIGHT_WRIST].y * h)
            },
            "INDEX": {
                "x": int(lm.landmark[lmPose.RIGHT_INDEX].x * w), 
                "y": int(lm.landmark[lmPose.RIGHT_INDEX].y * h)
            },
            "HIP": {
                "x": int(lm.landmark[lmPose.RIGHT_HIP].x * w), 
                "y": int(lm.landmark[lmPose.RIGHT_HIP].y * h)
            },
            "KNEE": {
                "x": int(lm.landmark[lmPose.RIGHT_KNEE].x * w), 
                "y": int(lm.landmark[lmPose.RIGHT_KNEE].y * h)
            },
            "HEEL": {
                "x": int(lm.landmark[lmPose.RIGHT_HEEL].x * w), 
                "y": int(lm.landmark[lmPose.RIGHT_HEEL].y * h)
            },
            "FOOT_INDEX": {
                "x": int(lm.landmark[lmPose.RIGHT_FOOT_INDEX].x * w), 
                "y": int(lm.landmark[lmPose.RIGHT_FOOT_INDEX].y * h)
            }
        }
    }
    return landmark

def getInclinations(landmark):
    LEFT, RIGHT = "LEFT", "RIGHT"
    inclinations = {
        "LEFT":{
            "NECK": findAngle(landmark[LEFT]["SHOULDER"]["x"], landmark[LEFT]["SHOULDER"]["y"], landmark[LEFT]["EAR"]["x"], landmark[LEFT]["EAR"]["y"]),
            "TORSO": findAngle(landmark[LEFT]["HIP"]["x"], landmark[LEFT]["HIP"]["y"], landmark[LEFT]["SHOULDER"]["x"], landmark[LEFT]["SHOULDER"]["y"]),
            "THIGH": findAngle(landmark[LEFT]["HIP"]["x"], landmark[LEFT]["HIP"]["y"], landmark[LEFT]["KNEE"]["x"], landmark[LEFT]["KNEE"]["y"]),
            "CALF": findAngle(landmark[LEFT]["KNEE"]["x"], landmark[LEFT]["KNEE"]["y"], landmark[LEFT]["HEEL"]["x"], landmark[LEFT]["HEEL"]["y"]),
            "ANKLE": findAngle(landmark[LEFT]["HEEL"]["x"], landmark[LEFT]["HEEL"]["y"], landmark[LEFT]["FOOT_INDEX"]["x"], landmark[LEFT]["FOOT_INDEX"]["y"]),
            "UPPERARM": findAngle(landmark[LEFT]["SHOULDER"]["x"], landmark[LEFT]["SHOULDER"]["y"], landmark[LEFT]["ELBOW"]["x"], landmark[LEFT]["ELBOW"]["y"]),
            "FOREARM": findAngle(landmark[LEFT]["ELBOW"]["x"], landmark[LEFT]["ELBOW"]["y"], landmark[LEFT]["WRIST"]["x"], landmark[LEFT]["WRIST"]["y"]),
            "WRIST": findAngle(landmark[LEFT]["WRIST"]["x"], landmark[LEFT]["WRIST"]["y"], landmark[LEFT]["INDEX"]["x"], landmark[LEFT]["INDEX"]["y"])
        },
        "RIGHT":{
            "NECK": findAngle(landmark[RIGHT]["SHOULDER"]["x"], landmark[RIGHT]["SHOULDER"]["y"], landmark[RIGHT]["EAR"]["x"], landmark[RIGHT]["EAR"]["y"]),
            "TORSO": findAngle(landmark[RIGHT]["HIP"]["x"], landmark[RIGHT]["HIP"]["y"], landmark[RIGHT]["SHOULDER"]["x"], landmark[RIGHT]["SHOULDER"]["y"]),
            "THIGH": findAngle(landmark[RIGHT]["HIP"]["x"], landmark[RIGHT]["HIP"]["y"], landmark[RIGHT]["KNEE"]["x"], landmark[RIGHT]["KNEE"]["y"]),
            "CALF": findAngle(landmark[RIGHT]["KNEE"]["x"], landmark[RIGHT]["KNEE"]["y"], landmark[RIGHT]["HEEL"]["x"], landmark[RIGHT]["HEEL"]["y"]),
            "ANKLE": findAngle(landmark[RIGHT]["HEEL"]["x"], landmark[RIGHT]["HEEL"]["y"], landmark[RIGHT]["FOOT_INDEX"]["x"], landmark[RIGHT]["FOOT_INDEX"]["y"]),
            "UPPERARM": findAngle(landmark[RIGHT]["SHOULDER"]["x"], landmark[RIGHT]["SHOULDER"]["y"], landmark[RIGHT]["ELBOW"]["x"], landmark[RIGHT]["ELBOW"]["y"]),
            "FOREARM": findAngle(landmark[RIGHT]["ELBOW"]["x"], landmark[RIGHT]["ELBOW"]["y"], landmark[RIGHT]["WRIST"]["x"], landmark[RIGHT]["WRIST"]["y"]),
            "WRIST": findAngle(landmark[RIGHT]["WRIST"]["x"], landmark[RIGHT]["WRIST"]["y"], landmark[RIGHT]["INDEX"]["x"], landmark[RIGHT]["INDEX"]["y"])
        }
    }
    return inclinations


def connectLandmarks(landmark, color):
    def connectDots(SIDE, landmark, color):
        #upper body
        cv2.line(image, (landmark[SIDE]["SHOULDER"]["x"], landmark[SIDE]["SHOULDER"]["y"]), (landmark[SIDE]["EAR"]["x"], landmark[SIDE]["EAR"]["y"]), color, 4)
        cv2.line(image, (landmark[SIDE]["HIP"]["x"], landmark[SIDE]["HIP"]["y"]), (landmark[SIDE]["SHOULDER"]["x"], landmark[SIDE]["SHOULDER"]["y"]), color, 4)

        #arms
        cv2.line(image, (landmark[SIDE]["ELBOW"]["x"], landmark[SIDE]["ELBOW"]["y"]), (landmark[SIDE]["SHOULDER"]["x"], landmark[SIDE]["SHOULDER"]["y"]), color, 4)
        cv2.line(image, (landmark[SIDE]["ELBOW"]["x"], landmark[SIDE]["ELBOW"]["y"]), (landmark[SIDE]["WRIST"]["x"], landmark[SIDE]["WRIST"]["y"]), color, 4)
        cv2.line(image, (landmark[SIDE]["WRIST"]["x"], landmark[SIDE]["WRIST"]["y"]), (landmark[SIDE]["INDEX"]["x"], landmark[SIDE]["INDEX"]["y"]), color, 4)

        #legs
        cv2.line(image, (landmark[SIDE]["KNEE"]["x"], landmark[SIDE]["KNEE"]["y"]), (landmark[SIDE]["HIP"]["x"], landmark[SIDE]["HIP"]["y"]), color, 4)
        cv2.line(image, (landmark[SIDE]["HEEL"]["x"], landmark[SIDE]["HEEL"]["y"]), (landmark[SIDE]["KNEE"]["x"], landmark[SIDE]["KNEE"]["y"]), color, 4)
        cv2.line(image, (landmark[SIDE]["FOOT_INDEX"]["x"], landmark[SIDE]["FOOT_INDEX"]["y"]), (landmark[SIDE]["HEEL"]["x"], landmark[SIDE]["HEEL"]["y"]), color, 4)

        # reference vertical lines
        cv2.line(image, (landmark[SIDE]["SHOULDER"]["x"], landmark[SIDE]["SHOULDER"]["y"]), (landmark[SIDE]["SHOULDER"]["x"], landmark[SIDE]["SHOULDER"]["y"] - 100), blue, 4)
        cv2.line(image, (landmark[SIDE]["HIP"]["x"], landmark[SIDE]["HIP"]["y"]), (landmark[SIDE]["HIP"]["x"], landmark[SIDE]["HIP"]["y"] - 100), blue, 4)
        cv2.line(image, (landmark[SIDE]["KNEE"]["x"], landmark[SIDE]["KNEE"]["y"]), (landmark[SIDE]["KNEE"]["x"], landmark[SIDE]["KNEE"]["y"] - 100), blue, 4)
    
    LEFT, RIGHT = "LEFT", "RIGHT"
    
    #connectors
    cv2.line(image, (landmark[LEFT]["SHOULDER"]["x"], landmark[LEFT]["SHOULDER"]["y"]), (landmark[RIGHT]["SHOULDER"]["x"], landmark[RIGHT]["SHOULDER"]["y"]), color, 4)
    cv2.line(image, (landmark[LEFT]["HIP"]["x"], landmark[LEFT]["HIP"]["y"]), (landmark[RIGHT]["HIP"]["x"], landmark[RIGHT]["HIP"]["y"]), color, 4)
    
    connectDots(LEFT, landmark, color)
    connectDots(RIGHT, landmark, color)


def drawLandmarks(landmark):
    def drawSide(SIDE, landmark, color):
        cv2.circle(image, (landmark[SIDE]["EAR"]["x"], landmark[SIDE]["EAR"]["y"]), 7, color, -1)
        cv2.circle(image, (landmark[SIDE]["SHOULDER"]["x"], landmark[SIDE]["SHOULDER"]["y"]), 7, color, -1)
        cv2.circle(image, (landmark[SIDE]["ELBOW"]["x"], landmark[SIDE]["ELBOW"]["y"]), 7, color, -1)
        cv2.circle(image, (landmark[SIDE]["WRIST"]["x"], landmark[SIDE]["WRIST"]["y"]), 7, color, -1)
        cv2.circle(image, (landmark[SIDE]["INDEX"]["x"], landmark[SIDE]["INDEX"]["y"]), 7, color, -1)
        cv2.circle(image, (landmark[SIDE]["HIP"]["x"], landmark[SIDE]["HIP"]["y"]), 7, color, -1)
        cv2.circle(image, (landmark[SIDE]["KNEE"]["x"], landmark[SIDE]["KNEE"]["y"]), 7, color, -1)
        cv2.circle(image, (landmark[SIDE]["HEEL"]["x"], landmark[SIDE]["HEEL"]["y"]), 7, color, -1)
        cv2.circle(image, (landmark[SIDE]["FOOT_INDEX"]["x"], landmark[SIDE]["FOOT_INDEX"]["y"]), 7, color, -1)

    LEFT, RIGHT = "LEFT", "RIGHT"

    drawSide(LEFT, landmark, yellow)
    drawSide(RIGHT, landmark, pink)


def displayAngles(inclinations, landmark, color):
    def displaySide(SIDE, landmark, color):
        cv2.putText(image, str(int(inclinations[SIDE]["NECK"])), (landmark[SIDE]["EAR"]["x"] + 10, landmark[SIDE]["EAR"]["y"]), font, 0.9, color, 2)
        cv2.putText(image, str(int(inclinations[SIDE]["TORSO"])), (landmark[SIDE]["SHOULDER"]["x"] + 10, landmark[SIDE]["SHOULDER"]["y"]), font, 0.9, color, 2)
        cv2.putText(image, str(int(inclinations[SIDE]["THIGH"])), (landmark[SIDE]["HIP"]["x"] + 10, landmark[SIDE]["HIP"]["y"]), font, 0.9, color, 2)
        cv2.putText(image, str(int(inclinations[SIDE]["CALF"])), (landmark[SIDE]["KNEE"]["x"] + 10, landmark[SIDE]["KNEE"]["y"]), font, 0.9, color, 2)
        cv2.putText(image, str(int(inclinations[SIDE]["ANKLE"])), (landmark[SIDE]["HEEL"]["x"] + 10, landmark[SIDE]["HEEL"]["y"]), font, 0.9, color, 2)
        cv2.putText(image, str(int(inclinations[SIDE]["UPPERARM"])), (landmark[SIDE]["ELBOW"]["x"] + 10, landmark[SIDE]["ELBOW"]["y"]), font, 0.9, color, 2)
        cv2.putText(image, str(int(inclinations[SIDE]["FOREARM"])), (landmark[SIDE]["WRIST"]["x"] + 10, landmark[SIDE]["WRIST"]["y"]), font, 0.9, color, 2)
        cv2.putText(image, str(int(inclinations[SIDE]["WRIST"])), (landmark[SIDE]["INDEX"]["x"] + 10, landmark[SIDE]["INDEX"]["y"]), font, 0.9, color, 2)

    LEFT, RIGHT = "LEFT", "RIGHT"

    displaySide(LEFT, landmark, color)
    displaySide(RIGHT, landmark, color)

"""
Function to send alert. Use this function to send alert when bad posture detected.
Feel free to get creative and customize as per your convenience.
"""

def sendWarning(x):
    pass


# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initilize frame counters.
good_frames = 0
bad_frames = 0

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# ===============================================================================================#


if __name__ == "__main__":
    # For webcam input replace file name with 0.
    file_name = 0 #'input1.mp4'
    cap = cv2.VideoCapture(file_name)

    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Video writer.
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

    while cap.isOpened():
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break
        # Get fps.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        h, w = image.shape[:2]

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image.
        keypoints = pose.process(image)

        # Convert the image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        # Acquire the landmark coordinates.
        # Once aligned properly, left or right should not be a concern.      
        # Left shoulder.
        landmark = getLandmarks(lm, w, h)

        # l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        # l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        # # Right shoulder
        # r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        # r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        # # Left elbow
        # l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
        # l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
        # # Left wrist
        # l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
        # l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
        # # Left ear.
        # l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        # l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        # # Left hip.
        # l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        # l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
        # # Right hip.
        # r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
        # r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)
        # # Left knee
        # l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
        # l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)
        # # Left ankle.
        # l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
        # l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)

        # Calculate distance between left shoulder and right shoulder points.
        # offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        offset = findDistance(landmark["LEFT"]["SHOULDER"]["x"], landmark["LEFT"]["SHOULDER"]["y"], landmark["RIGHT"]["SHOULDER"]["x"], landmark["RIGHT"]["SHOULDER"]["y"])

        # Assist to align the camera to point at the side view of the person.
        # Offset threshold 30 is based on results obtained from analysis over 100 samples.
        if offset < 100:
            cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
        else:
            cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

        # Calculate angles.
        # neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        # torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
        # thigh_inclination = findAngle(l_hip_x, l_hip_y, l_knee_x, l_knee_y)
        # calf_inclination = findAngle(l_knee_x, l_knee_y, l_ankle_x, l_ankle_y)
        # upper_arm_inclination = findAngle(l_shldr_x, l_shldr_y, l_elbow_x, l_elbow_y)
        # lower_arm_inclination = findAngle(l_elbow_x, l_elbow_y, l_wrist_x, l_wrist_y)
        inclinations = getInclinations(landmark)
        # neck_inclination = findAngle(landmark["LEFT"]["SHOULDER"]["x"], landmark["LEFT"]["SHOULDER"]["y"], landmark["LEFT"]["EAR"]["x"], landmark["LEFT"]["EAR"]["y"])
        # torso_inclination = findAngle(landmark["LEFT"]["HIP"]["x"], landmark["LEFT"]["HIP"]["y"], landmark["LEFT"]["SHOULDER"]["x"], landmark["LEFT"]["SHOULDER"]["y"])
        # thigh_inclination = findAngle(landmark["LEFT"]["HIP"]["x"], landmark["LEFT"]["HIP"]["y"], landmark["LEFT"]["KNEE"]["x"], landmark["LEFT"]["KNEE"]["y"])
        # calf_inclination = findAngle(landmark["LEFT"]["KNEE"]["x"], landmark["LEFT"]["KNEE"]["y"], landmark["LEFT"]["HEEL"]["x"], landmark["LEFT"]["HEEL"]["y"])
        # upper_arm_inclination = findAngle(landmark["LEFT"]["SHOULDER"]["x"], landmark["LEFT"]["SHOULDER"]["y"], landmark["LEFT"]["ELBOW"]["x"], landmark["LEFT"]["ELBOW"]["y"])
        # lower_arm_inclination = findAngle(landmark["LEFT"]["ELBOW"]["x"], landmark["LEFT"]["ELBOW"]["y"], landmark["LEFT"]["WRIST"]["x"], landmark["LEFT"]["WRIST"]["y"])

        # Draw landmarks.
        drawLandmarks(landmark)
        # cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        # cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
        # cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        # cv2.circle(image, (landmark["RIGHT"]["EAR"]["x"], landmark["RIGHT"]["EAR"]["y"]), 7, pink, -1)

        # # Let's take y - coordinate of P3 100px above x1,  for display elegance.
        # # Although we are taking y = 0 while calculating angle between P1,P2,P3.
        # # cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        # # cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        # cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
        # cv2.circle(image, (r_hip_x, r_hip_y), 7, pink, -1)
        
        # # Similarly, here we are taking y - coordinate 100px above x1. Note that
        # # you can take any value for y, not necessarily 100 or 200 pixels.
        # # cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)
        # cv2.circle(image, (l_knee_x, l_knee_y), 7, yellow, -1)
        # cv2.circle(image, (landmark["RIGHT"]["KNEE"]["x"], landmark["RIGHT"]["KNEE"]["y"]), 7, pink, -1)
        # # cv2.circle(image, (l_ankle_x, l_ankle_y), 7, yellow, -1)
        # cv2.circle(image, (landmark["LEFT"]["HEEL"]["x"], landmark["LEFT"]["HEEL"]["y"]), 7, yellow, -1)
        # cv2.circle(image, (landmark["RIGHT"]["HEEL"]["x"], landmark["RIGHT"]["HEEL"]["y"]), 7, pink, -1)

        # # arms
        # cv2.circle(image, (l_elbow_x, l_elbow_y), 7, yellow, -1)
        # cv2.circle(image, (landmark["RIGHT"]["ELBOW"]["x"], landmark["RIGHT"]["ELBOW"]["y"]), 7, pink, -1)
        # cv2.circle(image, (l_wrist_x, l_wrist_y), 7, yellow, -1)
        # cv2.circle(image, (landmark["RIGHT"]["WRIST"]["x"], landmark["RIGHT"]["WRIST"]["y"]), 7, pink, -1)

        # # feet
        # cv2.circle(image, (landmark["LEFT"]["FOOT_INDEX"]["x"], landmark["LEFT"]["FOOT_INDEX"]["y"]), 7, yellow, -1)
        # cv2.circle(image, (landmark["RIGHT"]["FOOT_INDEX"]["x"], landmark["RIGHT"]["FOOT_INDEX"]["y"]), 7, pink, -1)

        # # finger
        # cv2.circle(image, (landmark["LEFT"]["INDEX"]["x"], landmark["LEFT"]["INDEX"]["y"]), 7, yellow, -1)
        # cv2.circle(image, (landmark["RIGHT"]["INDEX"]["x"], landmark["RIGHT"]["INDEX"]["y"]), 7, pink, -1)

        # Put text, Posture and angle inclination.
        # Text string for display.
        angle_text_string_upper_body = 'Neck : ' + str(int(inclinations["LEFT"]["NECK"])) + '  Torso : ' + str(int(inclinations["LEFT"]["TORSO"]))
        angle_text_string_lower_body = 'Upper Leg : ' + str(int(inclinations["LEFT"]["THIGH"])) + '  Lower Leg : ' + str(int(inclinations["LEFT"]["CALF"]))
        angle_text_string_arms = 'Upper Arm : ' + str(int(inclinations["LEFT"]["UPPERARM"])) + '  Lower Arm : ' + str(int(inclinations["LEFT"]["FOREARM"]))

        # Determine whether good posture or bad posture.
        # The threshold angles have been set based on intuition.
        if 110 < inclinations["LEFT"]["NECK"] < 130 \
            and 110 < inclinations["LEFT"]["TORSO"] < 130 \
            and 130 < inclinations["LEFT"]["THIGH"] < 150 \
            and 140 < inclinations["LEFT"]["CALF"] < 160 \
            and 150 < inclinations["LEFT"]["UPPERARM"] < 170 \
            and 140 < inclinations["LEFT"]["FOREARM"] < 160:
            bad_frames = 0
            good_frames += 1
            
            cv2.putText(image, angle_text_string_upper_body, (10, 30), font, 0.5, light_green, 2)
            cv2.putText(image, angle_text_string_lower_body, (10, 60), font, 0.5, light_green, 2)
            cv2.putText(image, angle_text_string_arms, (10, 90), font, 0.5, light_green, 2)

            displayAngles(inclinations, landmark, light_green)
            # cv2.putText(image, str(int(inclinations["NECK"])), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
            # cv2.putText(image, str(int(inclinations["TORSO"])), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)
            # cv2.putText(image, str(int(inclinations["THIGH"])), (l_knee_x + 10, l_knee_y), font, 0.9, light_green, 2)
            # # cv2.putText(image, str(int(calf_inclination)), (l_ankle_x + 10, l_ankle_y), font, 0.9, light_green, 2)
            # cv2.putText(image, str(int(inclinations["CALF"])), (landmark["LEFT"]["HEEL"]["x"] + 10, landmark["LEFT"]["HEEL"]["y"]), font, 0.9, light_green, 2)
            # cv2.putText(image, str(int(inclinations["UPPERARM"])), (l_elbow_x + 10, l_elbow_y), font, 0.9, light_green, 2)
            # cv2.putText(image, str(int(inclinations["FOREARM"])), (l_wrist_x + 10, l_wrist_y), font, 0.9, light_green, 2)

            # Join landmarks.
            connectLandmarks(landmark, green)

        else:
            good_frames = 0
            bad_frames += 1

            cv2.putText(image, angle_text_string_upper_body, (10, 30), font, 0.5, red, 2)
            cv2.putText(image, angle_text_string_lower_body, (10, 60), font, 0.5, red, 2)
            cv2.putText(image, angle_text_string_arms, (10, 90), font, 0.5, red, 2)

            displayAngles(inclinations, landmark, red)
            # cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
            # cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)
            # cv2.putText(image, str(int(thigh_inclination)), (l_knee_x + 10, l_knee_y), font, 0.9, red, 2)
            # # cv2.putText(image, str(int(calf_inclination)), (l_ankle_x + 10, l_ankle_y), font, 0.9, red, 2)
            # cv2.putText(image, str(int(calf_inclination)), (landmark["LEFT"]["HEEL"]["x"] + 10, landmark["LEFT"]["HEEL"]["y"]), font, 0.9, red, 2)
            # cv2.putText(image, str(int(upper_arm_inclination)), (l_elbow_x + 10, l_elbow_y), font, 0.9, red, 2)
            # cv2.putText(image, str(int(lower_arm_inclination)), (l_wrist_x + 10, l_wrist_y), font, 0.9, red, 2)

            # Join landmarks.
            connectLandmarks(landmark, red)

        # Calculate the time of remaining in a particular posture.
        good_time = (1 / fps) * good_frames
        bad_time =  (1 / fps) * bad_frames

        # Pose time.
        if good_time > 0:
            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
            cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
            cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

        # If you stay in bad posture for more than 3 minutes (180s) send an alert.
        if bad_time > 180:
            sendWarning()
        # Write frames.
        video_output.write(image)

        # Display.
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
