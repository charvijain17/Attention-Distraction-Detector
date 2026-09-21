import cv2, mediapipe as mp, numpy as np, time, csv
from datetime import datetime

mp_face_mesh = mp.solutions.face_mesh

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]

# ------------------ EAR FUNCTION ------------------
def ear(landmarks, eye, w, h):
    pts = np.array([(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in eye])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)

# ------------------ PARAMETERS ------------------
CALIB_SECONDS = 2.0        # eyes open for calibration
BLINK_FILTER_SEC = 1.2     # for drowsiness
DISTRACT_FILTER_SEC = 1.0  # head turned for this long
LOG_EVERY_SEC = 60.0       # log every 1 minute

cap = cv2.VideoCapture(0)

# create CSV
with open("log.csv", "w", newline="") as f:
    csv.writer(f).writerow(["timestamp", "status"])

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as fm:

    # Calibration
    t0 = time.time()
    ear_sum = 0.0
    ear_cnt = 0
    thr = 0.25
    calibrated = False

    # State
    status = "Unknown"
    below_since = None
    distract_since = None
    last_face_time = time.time()
    last_log_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        have_face = res.multi_face_landmarks is not None

        if have_face:
            last_face_time = time.time()

            fl = res.multi_face_landmarks[0].landmark

            # EAR
            e_left = ear(fl, LEFT_EYE, w, h)
            e_right = ear(fl, RIGHT_EYE, w, h)
            e = (e_left + e_right) / 2.0

            # ---------- Calibration ----------
            if not calibrated:
                if time.time() - t0 < CALIB_SECONDS:
                    ear_sum += e
                    ear_cnt += 1
                    cv2.putText(frame, "Calibrating... Keep eyes OPEN",
                                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                else:
                    if ear_cnt > 10:
                        open_base = ear_sum / ear_cnt
                        thr = max(0.18, min(0.32, open_base * 0.75))
                    calibrated = True

            # ---------- Blink / Drowsy Filter ----------
            if e < thr:
                below_since = below_since or time.time()
            else:
                below_since = None

            # ---------- Head Position for DISTRACTION ----------
            nose_x = fl[1].x  # nose landmark
            center_tolerance = 0.15  # range for center

            if nose_x < 0.5 - center_tolerance or nose_x > 0.5 + center_tolerance:
                distract_since = distract_since or time.time()
            else:
                distract_since = None

            # ---------- STATUS DECISION ----------
            if not calibrated:
                status = "Calibrating"
                color = (0, 255, 255)

            else:
                if below_since and time.time() - below_since >= BLINK_FILTER_SEC:
                    status = "Drowsy"
                    color = (0, 0, 255)

                elif distract_since and time.time() - distract_since >= DISTRACT_FILTER_SEC:
                    status = "Distracted"
                    color = (255, 0, 0)

                else:
                    status = "Attentive"
                    color = (0, 255, 0)

            # Draw EAR values
            cv2.putText(frame, f"EAR: {e:.2f} Thr: {thr:.2f}",
                        (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        else:
            # No face for > 1 sec -> Unknown
            if time.time() - last_face_time > 1.0:
                status = "Unknown"
                color = (0, 255, 255)

        # ---------- BIG STATUS BANNER ----------
        cv2.putText(frame, f"STATUS: {status}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.imshow("Final Phase-1 Detector", frame)

        # ---------- CSV LOGGING ----------
        now = time.time()
        if now - last_log_time >= LOG_EVERY_SEC:
            last_log_time = now
            with open("log.csv", "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    status
                ])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
