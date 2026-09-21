import os
import cv2
import time
import csv
import json
import shutil
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageTk
import mediapipe as mp
import matplotlib

# use non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Appearance
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("dark-blue")

# Constants
mp_face_mesh = mp.solutions.face_mesh
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

PIE_COLORS = {
    "Attentive": "#2ecc71",
    "Distracted": "#e74c3c",
    "Drowsy": "#3498db",
    "Unknown": "#95a5a6"
}

# Utilities
def format_seconds(s):
    s = int(round(s))
    return str(timedelta(seconds=s))

def ensure_reports_dir():
    REPORTS_DIR.mkdir(exist_ok=True)

def compute_ear(landmarks, eye, w, h):
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def create_pie_chart(summary_seconds, out_path):
    labels = []
    sizes = []
    colors = []
    for k in ["Attentive","Distracted","Drowsy","Unknown"]:
        labels.append(k)
        sizes.append(summary_seconds.get(k, 0))
        colors.append(PIE_COLORS[k])
    if sum(sizes) == 0:
        sizes = [1,0,0,0]
    plt.figure(figsize=(4,4), dpi=100)
    plt.pie(sizes, labels=labels, autopct=lambda p: f"{p:.1f}%" if sum(sizes)>0 else "", colors=colors, startangle=90)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# App
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Driven Classroom Distraction Detector")
        self.root.geometry("1100x700")
        self.root.minsize(900,600)
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Frames
        self.title_frame = ctk.CTkFrame(root, height=100)
        self.title_frame.pack(fill="x", padx=12, pady=(12,6))
        self.title_label = ctk.CTkLabel(self.title_frame, text="AI Driven Classroom Distraction Detector", font=("Helvetica", 28, "bold"))
        self.title_label.pack(pady=18)

        self.content_frame = ctk.CTkFrame(root)
        self.content_frame.pack(fill="both", expand=True, padx=12, pady=6)

        self.left_frame = ctk.CTkFrame(self.content_frame, width=760)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=(0,8), pady=8)

        self.right_frame = ctk.CTkFrame(self.content_frame, width=320)
        self.right_frame.pack(side="right", fill="y", padx=(8,0), pady=8)

        # Right-side controls
        self.create_right_panel()

        # State
        self.cap = None
        self.mp_face = None
        self.detecting = False
        self.after_id = None
        self.video_label = None
        self.status_display = None
        self.report_log = []
        self.summary_seconds = {}
        self.current_status = "Unknown"
        self.last_status_change = time.time()

        # Report view holders
        self.reports_left_list = None
        self.reports_right_detail = None

        # Show home
        self.create_home_view()
        
    # Right panel
    def create_right_panel(self):
        for w in self.right_frame.winfo_children():
            w.destroy()
        hdr = ctk.CTkLabel(self.right_frame, text="Reports Manager", font=("Helvetica", 14, "bold"))
        hdr.pack(pady=(12,6))
        refresh_btn = ctk.CTkButton(self.right_frame, text="Refresh", command=self.refresh_reports_list)
        refresh_btn.pack(pady=6)
        del_all = ctk.CTkButton(self.right_frame, text="Delete ALL Reports", fg_color="#e74c3c", hover_color="#c0392b", command=self.delete_all_reports)
        del_all.pack(pady=6)
        info = ctk.CTkLabel(self.right_frame, text="Reports stored in ./reports/\nEach run saved separately", font=("Helvetica", 10))
        info.pack(pady=8)

    # Home dashboard
    def create_home_view(self):
        for w in self.left_frame.winfo_children():
            w.destroy()

        container = ctk.CTkFrame(self.left_frame)
        container.pack(expand=True, fill="both", padx=20, pady=30)

        left_card = ctk.CTkFrame(container)
        left_card.pack(side="left", expand=True, fill="both", padx=12, pady=12)

        right_card = ctk.CTkFrame(container)
        right_card.pack(side="left", expand=True, fill="both", padx=12, pady=12)

        lbl1 = ctk.CTkLabel(left_card, text="START DETECTION", font=("Helvetica", 20, "bold"))
        lbl1.pack(pady=(40,10))
        sub1 = ctk.CTkLabel(left_card, text="Run live camera detection", font=("Helvetica", 12))
        sub1.pack(pady=(0,20))
        start_btn = ctk.CTkButton(left_card, text="Start Detection", width=240, height=60, font=("Helvetica", 14, "bold"), command=self.show_start_detection)
        start_btn.pack(pady=10)

        lbl2 = ctk.CTkLabel(right_card, text="REPORTS", font=("Helvetica", 20, "bold"))
        lbl2.pack(pady=(40,10))
        sub2 = ctk.CTkLabel(right_card, text="View saved reports", font=("Helvetica", 12))
        sub2.pack(pady=(0,20))
        reports_btn = ctk.CTkButton(right_card, text="Reports", width=240, height=60, font=("Helvetica", 14, "bold"), command=self.show_reports)
        reports_btn.pack(pady=10)

        note = ctk.CTkLabel(self.left_frame, text="Click a card to start detection or view reports", font=("Helvetica", 12))
        note.pack(pady=8)

    # Detection view (NO Back button here)
    def show_start_detection(self):
        for w in self.left_frame.winfo_children():
            w.destroy()

        # Video area (no back button)
        self.video_label = ctk.CTkLabel(self.left_frame, text="")
        self.video_label.pack(pady=6)

        # STOP (saves report)
        stop_btn = ctk.CTkButton(self.left_frame, text="STOP", width=240, height=64, fg_color="#e74c3c", hover_color="#c0392b", font=("Helvetica", 16, "bold"), command=self.stop_detection_from_ui)
        stop_btn.pack(pady=12)

        # status display
        self.status_display = ctk.CTkLabel(self.left_frame, text="Status: Calibrating...", font=("Helvetica", 13))
        self.status_display.pack(pady=4)

        self.start_camera_and_detection()

    def start_camera_and_detection(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap or not self.cap.isOpened():
            ctk.CTkLabel(self.left_frame, text="Cannot open camera", text_color="#e74c3c").pack()
            return

        self.detecting = True
        self.mp_face = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # detection params
        self.calib_seconds = 2.0
        self.blk_filter = 1.2
        self.dist_filter = 1.0

        self.t0 = time.time()
        self.ear_sum = 0.0
        self.ear_cnt = 0
        self.thr = 0.25
        self.calibrated = False

        self.current_status = "Unknown"
        self.last_status_change = time.time()
        self.summary_seconds = {"Attentive":0.0,"Distracted":0.0,"Drowsy":0.0,"Unknown":0.0}
        self.below_since = None
        self.distract_since = None
        self.last_face_time = time.time()
        self.report_log = []
        self._last_log_time = 0

        # start loop
        self._update_video_loop()

    def _update_video_loop(self):
        if not self.detecting:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after_id = self.root.after(100, self._update_video_loop)
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mp_face.process(rgb)
        have_face = res.multi_face_landmarks is not None
        status = self.current_status

        if have_face:
            self.last_face_time = time.time()
            fl = res.multi_face_landmarks[0].landmark
            e_left = compute_ear(fl, LEFT_EYE, w, h)
            e_right = compute_ear(fl, RIGHT_EYE, w, h)
            e = (e_left + e_right) / 2.0

            if not self.calibrated:
                if time.time() - self.t0 < self.calib_seconds:
                    self.ear_sum += e
                    self.ear_cnt += 1
                    status = "Calibrating"
                else:
                    if self.ear_cnt > 5:
                        base = self.ear_sum / self.ear_cnt
                        self.thr = max(0.18, min(0.32, base * 0.75))
                    self.calibrated = True
                    status = "Attentive"
            else:
                if e < self.thr:
                    self.below_since = self.below_since or time.time()
                else:
                    self.below_since = None
                nose_x = fl[1].x
                tol = 0.15
                if nose_x < 0.5 - tol or nose_x > 0.5 + tol:
                    self.distract_since = self.distract_since or time.time()
                else:
                    self.distract_since = None

                if self.below_since and time.time() - self.below_since >= self.blk_filter:
                    status = "Drowsy"
                elif self.distract_since and time.time() - self.distract_since >= self.dist_filter:
                    status = "Distracted"
                else:
                    status = "Attentive"
        else:
            if time.time() - self.last_face_time > 1.0:
                status = "Unknown"

        # banner color
        color = (0,255,0)
        if status == "Drowsy":
            color = (0,0,255)
        elif status == "Distracted":
            color = (231,76,60)
        elif status == "Unknown":
            color = (150,150,150)
        elif status == "Calibrating":
            color = (0,255,255)

        cv2.putText(frame, f"STATUS: {status}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.putText(frame, f"EAR Thr: {self.thr:.2f}", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        # update durations when status changes
        now = time.time()
        if status != self.current_status:
            elapsed = now - self.last_status_change
            self.summary_seconds[self.current_status] = self.summary_seconds.get(self.current_status,0) + elapsed
            self.current_status = status
            self.last_status_change = now

        # log every second
        if now - self._last_log_time >= 1.0:
            self.report_log.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status))
            self._last_log_time = now

        # show on UI
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((780,480))
        imgtk = ImageTk.PhotoImage(img)
        if self.video_label:
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk
        if self.status_display:
            self.status_display.configure(text=f"Status: {status}")

        # schedule next
        self.after_id = self.root.after(15, self._update_video_loop)

    # Stop and SAVE report
    def stop_detection_from_ui(self):
        # immediate stop & save
        self.detecting = False
        if getattr(self, "after_id", None):
            try:
                self.root.after_cancel(self.after_id)
            except:
                pass
            self.after_id = None

        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except:
            pass

        try:
            if self.mp_face:
                self.mp_face.close()
        except:
            pass

        # finalize last interval
        now = time.time()
        elapsed = now - self.last_status_change
        self.summary_seconds[self.current_status] = self.summary_seconds.get(self.current_status,0) + elapsed

        # save folder
        ensure_reports_dir()
        folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = REPORTS_DIR / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        # raw csv
        csv_path = folder / "log.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","status"])
            for r in self.report_log:
                w.writerow(r)

        # summary text (human-readable)
        summary_path = folder / "summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Report Time: {folder_name}\n\n")
            for k in ["Attentive","Distracted","Drowsy","Unknown"]:
                f.write(f"{k}: {format_seconds(self.summary_seconds.get(k,0))}\n")

        # meta.json (numeric seconds) for accurate display later
        meta_path = folder / "meta.json"
        meta = {
            "durations": {
                "Attentive": round(self.summary_seconds.get("Attentive",0), 2),
                "Distracted": round(self.summary_seconds.get("Distracted",0), 2),
                "Drowsy": round(self.summary_seconds.get("Drowsy",0), 2),
                "Unknown": round(self.summary_seconds.get("Unknown",0), 2),
            }
        }
        meta["total_time"] = round(sum(meta["durations"].values()), 2)
        try:
            with open(meta_path, "w") as mf:
                json.dump(meta, mf, indent=2)
        except Exception as e:
            print("meta save failed:", e)

        # pie
        pie_path = folder / "pie.png"
        try:
            create_pie_chart(self.summary_seconds, pie_path)
        except Exception as e:
            print("Pie generation failed:", e)

        # return to home and refresh
        for w in self.left_frame.winfo_children():
            w.destroy()
        self.create_home_view()
        self.refresh_reports_list()

        saved_lbl = ctk.CTkLabel(self.left_frame, text=f"Saved report: {folder_name}", text_color="#2ecc71")
        saved_lbl.pack(pady=6)

    # Reports list (Back to Home present here only)
    def show_reports(self):
        for w in self.left_frame.winfo_children():
            w.destroy()

        # Back to home button (present here)
        back_btn = ctk.CTkButton(self.left_frame, text="← Back to Home", width=180, height=40, command=self.create_home_view)
        back_btn.pack(pady=(6,6))

        hdr = ctk.CTkLabel(self.left_frame, text="Saved Reports", font=("Helvetica", 18, "bold"))
        hdr.pack(pady=8)

        split = ctk.CTkFrame(self.left_frame)
        split.pack(fill="both", expand=True, padx=10, pady=6)

        left_list = ctk.CTkFrame(split, width=300)
        left_list.pack(side="left", fill="y", padx=(0,8), pady=6)

        right_detail = ctk.CTkFrame(split)
        right_detail.pack(side="left", fill="both", expand=True, pady=6)

        self.reports_left_list = left_list
        self.reports_right_detail = right_detail

        scr = ctk.CTkScrollableFrame(left_list, width=300, height=520)
        scr.pack(fill="both", expand=True, padx=6, pady=6)

        folders = sorted([p for p in REPORTS_DIR.iterdir() if p.is_dir()], reverse=True)
        if not folders:
            lbl = ctk.CTkLabel(scr, text="No reports yet. Run detection to create one.")
            lbl.pack(pady=12)
            return

        for idx, f in enumerate(folders, start=1):
            name = f"Report {idx} - {f.name}"
            frame = ctk.CTkFrame(scr)
            frame.pack(fill="x", padx=6, pady=6)
            lbl = ctk.CTkLabel(frame, text=name, anchor="w")
            lbl.grid(row=0, column=0, sticky="w", padx=8)
            open_btn = ctk.CTkButton(frame, text="Open", width=70, command=lambda p=f: self.display_report_in_right(p))
            open_btn.grid(row=0, column=1, padx=6)
            del_btn = ctk.CTkButton(frame, text="Delete", width=70, fg_color="#e74c3c", hover_color="#c0392b", command=lambda p=f: (self.delete_report(p), self.show_reports()))
            del_btn.grid(row=0, column=2, padx=6)

            preview_text = ""
            sum_path = f / "summary.txt"
            if sum_path.exists():
                try:
                    with open(sum_path, "r") as s:
                        lines = s.readlines()
                        preview_text = lines[1].strip() if len(lines)>1 else ""
                except:
                    preview_text = ""
            preview = ctk.CTkLabel(frame, text=preview_text, anchor="w")
            preview.grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=4)

        # default open most recent
        if folders:
            self.display_report_in_right(folders[0])

    # Individual report view (NO "Back to Reports" button)
    def display_report_in_right(self, folder:Path):
        rd = self.reports_right_detail
        for w in rd.winfo_children():
            w.destroy()

        t = ctk.CTkLabel(rd, text=f"Report: {folder.name}", font=("Helvetica", 14, "bold"))
        t.pack(pady=8)

        # pie
        pie = folder / "pie.png"
        if pie.exists():
            try:
                pil = Image.open(pie).resize((360,360))
                tkp = ImageTk.PhotoImage(pil)
                img_lbl = ctk.CTkLabel(rd, image=tkp, text="")
                img_lbl.image = tkp
                img_lbl.pack(pady=6)
            except Exception:
                ctk.CTkLabel(rd, text="Could not load pie image").pack(pady=10)
        else:
            ctk.CTkLabel(rd, text="No pie chart available", font=("Helvetica", 12)).pack(pady=12)

        # load meta.json for numeric values
        meta_path = folder / "meta.json"
        total_time = None
        durations = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r") as mf:
                    meta = json.load(mf)
                    total_time = meta.get("total_time", None)
                    durations = meta.get("durations", {})
            except:
                durations = {}
        else:
            # Fallback: attempt to estimate from summary.txt by parsing formatted HH:MM:SS
            sum_path = folder / "summary.txt"
            durations = {}
            if sum_path.exists():
                try:
                    with open(sum_path, "r") as sf:
                        lines = sf.readlines()[1:]
                        for line in lines:
                            parts = line.strip().split(": ", 1)
                            if len(parts) == 2:
                                k = parts[0]
                                v = parts[1].strip()
                                # parse HH:MM:SS to seconds
                                try:
                                    h,m,s = [int(x) for x in v.split(":")]
                                    secs = h*3600 + m*60 + s
                                    durations[k] = secs
                                except:
                                    durations[k] = 0
                except:
                    pass
            total_time = round(sum(durations.values()), 2)

        # show numeric summary
        if total_time is None:
            # compute total_time from durations dict if possible
            total_time = round(sum(durations.get(k, 0) for k in ["Attentive","Distracted","Drowsy","Unknown"]), 2)

        # show textual summary under the pie chart
        text_frame = ctk.CTkFrame(rd)
        text_frame.pack(pady=6)
        # Total time (seconds) and formatted
        total_lbl = ctk.CTkLabel(text_frame, text=f"Total Detection Time: {total_time} sec ({format_seconds(total_time)})", font=("Helvetica", 12, "bold"))
        total_lbl.pack(pady=(4,2))

        # each status
        for key in ["Attentive","Distracted","Drowsy","Unknown"]:
            val = durations.get(key, 0)
            # if value looks like HH:MM:SS formatted in summary.txt conversion maybe already seconds
            try:
                display_secs = float(val)
            except:
                display_secs = 0.0
            l = ctk.CTkLabel(text_frame, text=f"{key}: {display_secs} sec ({format_seconds(display_secs)})", font=("Helvetica", 11))
            l.pack(anchor="w", padx=6)

        # summary text box (human readable)
        sum_path = folder / "summary.txt"
        txt = "No summary found."
        if sum_path.exists():
            try:
                with open(sum_path, "r") as f:
                    txt = f.read()
            except:
                txt = "Error reading summary."

        tb = ctk.CTkTextbox(rd, width=360, height=120)
        tb.pack(pady=6)
        tb.insert("0.0", txt)
        tb.configure(state="disabled")

        # actions (only delete)
        act = ctk.CTkFrame(rd)
        act.pack(pady=6)
        del_btn = ctk.CTkButton(act, text="Delete Report", fg_color="#e74c3c", hover_color="#c0392b", command=lambda p=folder: (self.delete_report(p), self.show_reports()))
        del_btn.grid(row=0, column=0, padx=6)

    # Reports helpers
    def refresh_reports_list(self):
        if self.reports_left_list is not None:
            self.show_reports()

    def delete_report(self, folder:Path):
        if folder.exists() and folder.is_dir():
            shutil.rmtree(folder)

    def delete_all_reports(self):
        if REPORTS_DIR.exists():
            for child in REPORTS_DIR.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
        self.refresh_reports_list()

    # Shutdown
    def on_closing(self):
        try:
            self.detecting = False
            if getattr(self, "after_id", None):
                try:
                    self.root.after_cancel(self.after_id)
                except:
                    pass
            try:
                if self.cap and self.cap.isOpened():
                    self.cap.release()
            except:
                pass
            try:
                if getattr(self, "mp_face", None):
                    self.mp_face.close()
            except:
                pass
        finally:
            try:
                self.root.destroy()
            except:
                pass

# Run
if __name__ == "__main__":
    root = ctk.CTk()
    app = App(root)
    root.mainloop()
