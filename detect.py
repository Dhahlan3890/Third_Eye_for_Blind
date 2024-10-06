import cv2
from ultralytics import YOLO
import pyttsx3
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

model = YOLO('best.pt')

cap = cv2.VideoCapture(0)
# cap.open("http://192.168.8.153:8080/video")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

print(model.names)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break
    
    results = model(frame)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    positions = []

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(class_id)

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        class_name = model.names[class_id]
        
        
        if (x1+x2)/2 > (2 * width)/3:
            object_width = x2 - x1
            position = "right"
            if object_width > 100:
                object_width = "large"
                positions.append(f"{position}_{object_width}")
                speak(f"There is a {class_name} at {position} side")

        elif (x1+x2)/2 < (width)/3:
            object_width = x2 - x1
            position = "left"
            if object_width > 100:
                object_width = "large"
                positions.append(f"{position}_{object_width}")
                speak(f"There is a {class_name} at {position} side")
            
        else:
            object_width = x2 - x1
            position = "center"
            if object_width > 100:
                object_width = "large"
                positions.append(f"{position}{object_width}")
                speak(f"There is a {class_name} infront of you careful")

        label = f"{class_name} ({confidence:.2f}) at {position}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    print(positions)
    # if len(positions) > 0:
    #     if "right_large" in positions:
    #         print("Alert! There is a large object on your right.")
    #         speak("Alert! Move to the left.")
    #     elif "left_large" in positions:
    #         print("Alert! There is a large object on your left.")
    #         speak("Alert! Move to the right.") 
    #     elif "center_large" in positions:
    #         print("Alert! There is a large object in front of you.")
    #         speak("Alert! There is a large object in front of you. Turn around.")
    
    cv2.imshow('Real-time Detection for blind people', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()