import tkinter as tk
from tkinter import messagebox
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import os
from src.tts import speak

MODEL_PATH = 'model/sign_model.h5'
IMG_SIZE = 28
LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

class SignLangApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator")
        self.root.geometry("900x600")
        self.root.minsize(400, 300)
        self.running = False
        self.prev_letter = None

        # Error handling for model
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error", f"Model file not found at {MODEL_PATH}. Please train the model first.")
            self.root.destroy()
            return

        self.model = load_model(MODEL_PATH)

        # UI
        self.label = tk.Label(root, text="Predicted Letter: ", font=("Arial", 24))
        self.label.pack(pady=10)
        self.btn = tk.Button(root, text="Start Detection", command=self.start_detection)
        self.btn.pack(pady=10)
        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.root.bind("<Configure>", self.on_resize)

        self.frame = None
        self.cap = None

    def start_detection(self):
        if not self.running:
            self.running = True
            self.btn.config(text="Stop Detection", command=self.stop_detection)
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Webcam not found. Please check your camera.")
                self.stop_detection()
                return
            threading.Thread(target=self.detect, daemon=True).start()

    def stop_detection(self):
        self.running = False
        self.btn.config(text="Start Detection", command=self.start_detection)
        if self.cap:
            self.cap.release()
            self.cap = None

    def detect(self):
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
            x1, y1, x2, y2 = 100, 100, 228, 228
            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=(0, -1))
            pred = self.model.predict(img)
            idx = np.argmax(pred)
            letter = LABELS[idx]
            self.label.config(text=f"Predicted Letter: {letter}")

            # Text-to-speech
            if letter != self.prev_letter:
                speak(letter)
                self.prev_letter = letter

            # Draw ROI rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            self.frame = frame.copy()
            self.update_canvas()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_canvas(self):
        if self.frame is not None:
            # Resize frame to fit canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((canvas_width, canvas_height), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def on_resize(self, event):
        self.update_canvas()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLangApp(root)
    root.mainloop()
