import cv2
import os
from datetime import datetime

class VideoCapture:
    def __init__(self):
        self.cap = None
        self.recording = False
        self.out = None
        self.output_dir = "captured_videos"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def start(self):
        self.cap = cv2.VideoCapture(0)
        self.recording = True
        self._start_recording()
    
    def _start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"recording_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(filename, fourcc, 20.0, (640,480))
    
    def get_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.recording:
                    self.out.write(frame)
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def stop_video(self):
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        self.recording = False
