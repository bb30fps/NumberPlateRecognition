import tkinter as tk
from tkinter import ttk, messagebox
from PIL import ImageTk, Image
import cv2
import torch
import yaml
from database import DatabaseManager
from video_capture import VideoCapture
from auth import AuthManager
from models.model import PlateRecognitionModel
import threading

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Secure Plate Recognition")
        self.geometry("1200x800")
        
        # Load configuration
        with open("utils/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize subsystems
        self.db = DatabaseManager()
        self.auth = AuthManager(self.db)
        self.video = VideoCapture()
        
        #User-Interface
        self.current_user = None
        self.transform = A.Compose([
            A.Resize(*self.config['image_size']),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        #load-model
        self.model = None
        try:
            self.model = PlateRecognitionModel(num_chars=len(self.config['chars']))
            checkpoint = torch.load("models/number_plate_model.pth")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
        
        # Show login screen first
        self.show_login()

    def show_login(self):
        # Login frame
        self.login_frame = ttk.Frame(self)
        
        ttk.Label(self.login_frame, text="Username:").grid(row=0, column=0)
        self.username_entry = ttk.Entry(self.login_frame)
        self.username_entry.grid(row=0, column=1)
        
        ttk.Label(self.login_frame, text="Password:").grid(row=1, column=0)
        self.password_entry = ttk.Entry(self.login_frame, show="*")
        self.password_entry.grid(row=1, column=1)
        
        ttk.Button(self.login_frame, text="Login", 
                  command=self.handle_login).grid(row=2, column=1)
        
        self.login_frame.pack(expand=True)

    def handle_login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        result = self.db.authenticate_user(username, password)
        
        if result:
            self.current_user = {
                'username': username,
                'is_admin': result[0]
            }
            self.db.log_action(username, "Login")
            self.show_main_app()
        else:
            messagebox.showerror("Error", "Invalid credentials")

    def show_main_app(self):
        # Destroy login frame
        self.login_frame.destroy()
        
        # Main application layout
        self.video_frame = ttk.LabelFrame(self, text="Live Camera Feed")
        self.controls_frame = ttk.Frame(self)
        
        # Video display
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()
        
        # Controls
        self.btn_start = ttk.Button(self.controls_frame, text="Start Camera",
                                   command=self.start_video)
        self.btn_stop = ttk.Button(self.controls_frame, text="Stop Camera",
                                  command=self.stop_video)
        self.btn_predict = ttk.Button(self.controls_frame, text="Run Detection",
                                     command=self.run_detection)
        
        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop.pack(side=tk.LEFT)
        self.btn_predict.pack(side=tk.LEFT)
        
        # Admin panel
        if self.auth.validate_permissions(self.current_user['username'], "admin"):
            self.admin_panel = ttk.LabelFrame(self, text="Admin Controls")
            ttk.Button(self.admin_panel, text="View Logs",
                      command=self.show_logs).pack()
            self.admin_panel.pack(fill=tk.X)
        
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        self.controls_frame.pack(fill=tk.X)

    def start_video(self):
        self.video.start()
        self.update_video_feed()

    def update_video_feed(self):
        frame = self.video.get_frame()
        if frame is not None:
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.after(10, self.update_video_feed)

    def run_detection(self):
        if not self.auth.validate_permissions(self.current_user['username'], "admin"):
            messagebox.showwarning("Permission Denied", 
                                 "Contact administrator for access")
            return
        
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return
        
        frame = self.video.get_frame()
        if frame is None:
            return
        
        try:
            # Preprocess frame
            transformed = self.transform(image=frame)
            img_tensor = transformed['image'].unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
                _, preds = torch.max(outputs, 2)
                plate_text = ''.join([self.config['chars'][i] for i in preds[0] if i < len(self.config['chars'])])
            
            # Display results
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f"Plate: {plate_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            # Log action
            self.db.log_action(self.current_user['username'], 
                             f"Detected plate: {plate_text}")
            
        except Exception as e:
            messagebox.showerror("Detection Error", str(e))

    def show_logs(self):
        logs_window = tk.Toplevel(self)
        logs_window.title("Activity Logs")
        
        # Create treeview
        tree = ttk.Treeview(logs_window, columns=("Timestamp", "User", "Action"))
        tree.heading("#0", text="ID")
        tree.heading("Timestamp", text="Timestamp")
        tree.heading("User", text="User")
        tree.heading("Action", text="Action")
        
        # Fetch logs
        c = self.db.conn.cursor()
        c.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 100")
        
        for log in c.fetchall():
            tree.insert("", tk.END, 
                        text=str(log[0]), 
                        values=(log[1], log[2], log[3]))
        
        tree.pack(fill=tk.BOTH, expand=True)

    def stop_video(self):
        self.video.stop_video()
        self.db.log_action(self.current_user['username'], "Stopped camera")

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
