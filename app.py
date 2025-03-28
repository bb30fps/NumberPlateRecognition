import tkinter as tk
from tkinter import ttk, messagebox
from PIL import ImageTk, Image
import cv2
from database import DatabaseManager
from video_capture import VideoCapture
from auth import AuthManager
import threading

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Secure Plate Recognition")
        self.geometry("1200x800")
        
        # Initialize subsystems
        self.db = DatabaseManager()
        self.auth = AuthManager(self.db)
        self.video = VideoCapture()
        
        # GUI State
        self.current_user = None
        self.is_admin = False
        
        # Show login screen first
        self.show_login()
    
    def show_login(self):
        # Login frame
        self.login_frame = ttk.Frame(self)
        
        ttk.Label(self.login_frame, text="Username:").grid(row=0, column=0)
        self.username = ttk.Entry(self.login_frame)
        self.username.grid(row=0, column=1)
        
        ttk.Label(self.login_frame, text="Password:").grid(row=1, column=0)
        self.password = ttk.Entry(self.login_frame, show="*")
        self.password.grid(row=1, column=1)
        
        ttk.Button(self.login_frame, text="Login", 
                  command=self.handle_login).grid(row=2, column=1)
        
        self.login_frame.pack(expand=True)
    
    def handle_login(self):
        username = self.username.get()
        password = self.password.get()
        result = self.db.authenticate_user(username, password)
        
        if result:
            self.current_user = username
            self.is_admin = result[0]
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
        if self.is_admin:
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
        if not self.is_admin:
            messagebox.showwarning("Permission Denied", 
                                  "Contact administrator for access")
            return
        
        # Add your model prediction logic here
        frame = self.video.get_frame()
        # processed_frame = model.predict(frame)
        # Update display with results
    
    def show_logs(self):
        # Implement log viewer
        pass

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
