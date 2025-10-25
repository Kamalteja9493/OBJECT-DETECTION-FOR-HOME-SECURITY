# 🏠 Object Detection for Home Security

This project implements real-time object detection for home security systems using YOLO models. It enables live webcam monitoring and sends alert emails when motion or specific objects are detected.

---

## 📁 Folder Contents

- **`run_project.sh`** – Main shell script to run detection and model comparison  
- **`webcam.py`** – Real-time webcam object detection with an alert email system  
- **`notebookce0b99c74d (1).py`** – Script to compare YOLOv3, YOLOv5, and YOLOv8 models on a custom dataset  
- **`yolov5m.pt`** – Pretrained YOLOv5 model used for real-time deployment  
- **`configuration.py`** – Stores email credentials and alert configurations  
- **`README.txt`** – Instruction guide (this file)  
- **`Other files/`** – Contains additional scripts and assets needed to run the project  

---

## 🛠 Requirements

Before running the project, install the required dependencies:

```bash
pip install ultralytics opencv-python tabulate
````

---

## ▶️ How to Run

1. **Give execute permission** to the shell script:

   ```bash
   chmod +x run_project.sh
   ```

2. **Run the project**:

   ```bash
   ./run_project.sh
   ```

3. All essential supporting files required for `run_project.sh` execution are available in the **`Other files/`** folder.

---

## 📧 Alert System

The script includes a built-in email alert system that:

* Detects objects in real-time via webcam.
* Sends an email notification with an image when an intruder or object of interest is detected.

Configuration details (email, password, SMTP setup) are defined in `configuration.py`.

---

## 🧠 Model Information

This project compares multiple YOLO versions:

* **YOLOv3**
* **YOLOv5**
* **YOLOv8**

It benchmarks their performance for home security object detection and uses the YOLOv5m model (`yolov5m.pt`) for deployment.

---

## 👨‍💻 Author

**Kamal Teja**
🔗 [GitHub Profile](https://github.com/Kamalteja9493)

---

````

---

