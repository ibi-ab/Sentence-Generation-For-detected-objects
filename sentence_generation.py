import cv2
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from openpose import pyopenpose as op

# Load YOLOv8 model
model = torch.hub.load('ultralytics/yolov8', 'yolov8s')

# Initialize OpenPose
params = {"model_folder": "models/"}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_sentence(prompt):
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt')
    outputs = gpt2_model.generate(inputs, max_length=50, num_return_sequences=1)
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def is_sitting(pose):
    # Custom logic to determine if the pose represents sitting
    # This is a simple heuristic: if the distance between the knee and the hip is less than a certain threshold, we consider the person to be sitting.
    # Note: You may need to refine this based on your specific requirements and data.
    try:
        hip_y = pose[11][1]  # Left hip y-coordinate
        knee_y = pose[12][1]  # Left knee y-coordinate
        if knee_y - hip_y < 50:  # Adjust the threshold based on your data
            return True
        return False
    except IndexError:
        return False

def is_walking(pose):
    # Custom logic to determine if the pose represents walking
    # Here we could check if the legs are apart and the person is not sitting
    try:
        left_hip = pose[11]  # Left hip coordinates
        right_hip = pose[8]  # Right hip coordinates
        left_knee = pose[12]  # Left knee coordinates
        right_knee = pose[9]  # Right knee coordinates
        # If the distance between hips is significant and knees are not bent as in sitting
        if abs(left_hip[0] - right_hip[0]) > 30 and not is_sitting(pose):
            return True
        return False
    except IndexError:
        return False

def process_frame(frame):
    # Object Detection
    results = model(frame)
    objects = results.xyxy[0]  # Bounding boxes

    # Pose Estimation
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    poses = datum.poseKeypoints

    # Scene Interpretation and Sentence Generation
    for obj in objects:
        if int(obj[5]) == 0:  # Class 0 for person
            for pose in poses:
                if is_sitting(pose):  # Custom function to check if sitting
                    prompt = "A person is sitting on a chair"
                    sentence = generate_sentence(prompt)
                    print(sentence)
                elif is_walking(pose):  # Custom function to check if walking
                    prompt = "A person is walking towards you"
                    sentence = generate_sentence(prompt)
                    print(sentence)

# Capture video and process frames
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    process_frame(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


