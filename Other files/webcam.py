from ultralytics import YOLO
import cv2
import smtplib
from email.message import EmailMessage
from configuration import EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER

# Load model
model = YOLO("/Users/kamalteja/Downloads/yolov5m.pt")

# Define "dangerous" class names
danger_classes = {"knife", "fire", "gun", "person"}  # ← adjust based on your use case
already_alerted = False

# Email alert function
def send_alert(detected_class):
    msg = EmailMessage()
    msg.set_content(f"️ ALERT: Detected {detected_class} on webcam!")
    msg["Subject"] = f"YOLOv5 ALERT: {detected_class.upper()} detected"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(msg)
    print(f"📧 Alert sent: {detected_class}")

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Webcam not accessible.")
else:
    print(" Webcam running. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model.predict(source=frame, conf=0.4, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            if cls_name in danger_classes and not already_alerted:
                send_alert(cls_name)
                already_alerted = True  # avoid repeated spamming

        # Show webcam with predictions
        results.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

