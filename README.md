# Number Plate Recognition System 🚗📄

A deep learning-based system to detect and recognize license plates from images using PyTorch. Trained on custom datasets with XML annotations.

# Features
- **Object Detection**: Localize license plates in images using bounding boxes.
- **Text Recognition**: Extract text from detected plates using a CNN + LSTM model.
- **Custom Dataset Support**: Works with XML and JPG annotations.
- **Google Colab Integration**: Train your model on GPU for faster results.

## Installation
1. **Clone the repository**:
   
   git clone https://github.com/bb30fps/NumberPlateRecognition.git
   
   cd number-plate-recognition


Here's a step-by-step manual to run your license plate recognition system in VS Code:

---

#### **1. Project Setup in VS Code**
1. **Folder Structure**:
   ```bash
   .
   ├── app.py
   ├── auth.py
   ├── database.py
   ├── requirements.txt
   ├── train.py
   ├── video_capture.py
   ├── models/
   │   └── model.py
   ├── utils/
   │   ├── config.yaml
   │   └── dataset.py
   ├── data/
   │   ├── annotations/  # Your XML files
   │   └── images/      # Your JPG/PNG images
   ```

## 2. Clone Repository
```bash
git clone https://github.com/yourusername/license-plate-recognition.git
cd license-plate-recognition
```

---

#### **2. Install Dependencies**
1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate Environment**:
   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```


3. **Install Packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

#### **3. Prepare Data**
1. **Dataset Requirements**:
   - Place images in `data/images/`
   - Place XML annotations in `data/annotations/`
   - Ensure XML files match image filenames:
     ```
     image_001.jpg ↔ image_001.xml
     ```

2. **Sample XML Structure**:
   ```xml
   <annotation>
     <filename>car_001.jpg</filename>
     <object>
       <name>ABC123</name>
       <bndbox>
         <xmin>100</xmin>
         <ymin>200</ymin>
         <xmax>300</xmax>
         <ymax>250</ymax>
       </bndbox>
     </object>
   </annotation>
   ```

---

#### **4. Initialize Database**
1. Create `database_init.py`:
   ```python
   from database import DatabaseManager

   db = DatabaseManager()
   db.add_user("admin", "admin123", is_admin=True)
   db.add_user("user", "user123")
   ```

2. Run initialization:
   ```bash
   python database_init.py
   ```

---

#### **5. Train the Model**
1. **Start Training**:
   ```bash
   python train.py
   ```

2. **Monitor Training**:
   - Loss values will print in terminal
   - Trained model saves to `models/number_plate_model.pth`

3. **Troubleshooting**:
   - If CUDA OOM: Reduce `batch_size` in `utils/config.yaml`
   - If XML errors: Validate annotation files

---

#### **6. Run the Application**
1. **Launch GUI**:
   ```bash
   python app.py
   ```

2. **Login Credentials**:
   ```
   Admin: username=admin / password=admin123
   User:  username=user  / password=user123
   ```

3. **Application Workflow**:
   ```
   1. Click "Start Camera"
   2. Position license plate in view
   3. Click "Run Detection" (admin only)
   4. View results in GUI
   5. Click "Stop Camera" to exit
   ```

---

#### **7. Key Features**
- **Admin Controls**:
  - Access detection functionality
  - View activity logs (implement in `show_logs()`)
- **User Tracking**:
  - All logins/actions recorded in `project.db`
- **Video Recording**:
  - Captures to `captured_videos/` folder

---

#### **Troubleshooting Guide**
| Issue | Solution |
|-------|----------|
| Webcam not working | Change `cv2.VideoCapture(0)` to `1` in `video_capture.py` |
| XML parsing errors | Validate files at [XML Validator](https://www.xmlvalidation.com/) |
| CUDA out of memory | Reduce `batch_size` in config.yaml |
| Database locked | Delete `project.db` and reinitialize |
| Import errors | Check Python path in VS Code (Ctrl+Shift+P > "Python: Select Interpreter") |

---

#### **Next Steps**
1. Implement log viewer in `show_logs()`
2. Add plate validation logic
3. Connect to license plate database API
4. Implement real-time video processing



Here's a step-by-step manual to set up and run your license plate recognition system in VS Code:

---

# **License Plate Recognition System Setup Guide**

## 1. Prerequisites
- **VS Code** installed
- **Python 3.8+** installed
- **Git** installed
- Webcam (for live detection)

## 2. Clone Repository
```bash
git clone https://github.com/yourusername/license-plate-recognition.git
cd license-plate-recognition
```

## 3. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

## 4. Install Dependencies
```bash
pip install -r requirement.txt
```

