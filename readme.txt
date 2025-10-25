# Object Detection for Home Security 


---

## Folder Contents:
- `run_project.sh` – Main shell script to run detection and model comparison
- `webcam.py` – Real-time webcam object detection with alert email system
- `notebookce0b99c74d (1).py` – Script to compare YOLOv3, YOLOv5, YOLOv8 on custom dataset
- `yolov5m.pt` – Trained YOLOv5 model used for real-time deployment
- `configuration.py` – Email credentials and alert config
- `README.txt` – This instructions file

---

## 🛠 Requirements:
Install the following dependencies before running:
```bash
pip install ultralytics opencv-python tabulate

#How to run:

chmod +x run_project.sh # Give execute permission to the .sh file in command terminal

./run_project.sh #to run the project

Remaining important files needed to run the run_project.sh are present in folder 'Other files'
