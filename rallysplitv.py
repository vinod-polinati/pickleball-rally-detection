import cv2
import numpy as np
import os
import subprocess
import sys
from ultralytics import YOLO

# ================= V18: THE GOLDILOCKS TUNE =================
VIDEO_PATH = "input_match.mp4"   
OUTPUT_DIR = "rallies_v18"
DEBUG_MODE = False                # <--- Run TRUE first
DEBUG_DURATION_LIMIT = 40        

# 1. GAP TOLERANCE (The "Merger" Fix)
# Lowered to 0.6s. If ball disappears for > 0.6s, FORCE NEW CLIP.
GAP_TOLERANCE_SEC = 0.6          

# 2. PHYSICS SENSITIVITY (The "Splitter" Fix)
# Lowered to 300px. 
# Smashes are ~100px. Cuts are ~800px. 
# 300px is the safe middle ground.
MAX_BALL_JUMP = 300.0            

# 3. SHOE FILTER (Green Shoe Protection)
FOOT_ZONE_RATIO = 0.45           

# 4. CONFIG
MIN_RALLY_DURATION = 1.0         
CONF_THRESHOLD = 0.15            
IMG_SIZE = 1280                  
# ============================================================

def get_ffmpeg_cmd(input_file, start, end, output_file):
    return [
        "ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", input_file, 
        "-t", f"{end-start:.2f}", "-c", "copy", "-loglevel", "error", output_file
    ]

def calculate_dist(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def get_vector(pt1, pt2):
    vec = np.array(pt2) - np.array(pt1)
    mag = np.linalg.norm(vec)
    if mag == 0: return np.zeros(2), 0
    return vec / mag, mag

def is_shoe(ball_box, person_boxes):
    bx1, by1, bx2, by2 = ball_box.xyxy[0].cpu().numpy()
    b_center_x = (bx1 + bx2) / 2
    b_center_y = (by1 + by2) / 2
    for p_box in person_boxes:
        px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy()
        p_h = py2 - py1
        if px1 < b_center_x < px2:
            foot_zone_top = py2 - (p_h * FOOT_ZONE_RATIO)
            if b_center_y > foot_zone_top and b_center_y < py2 + 40: 
                return True
    return False

def detect_rallies():
    print(f"🚀 Starting V18 (Goldilocks Tune)...")
    
    model = YOLO('yolov8x.pt') 
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Info: {w_frame}x{h_frame} @ {fps}FPS")

    debug_writer = None
    if DEBUG_MODE:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        debug_writer = cv2.VideoWriter("debug_v18.mp4", fourcc, fps, (w_frame, h_frame))

    ball_timeline = np.zeros(frames, dtype=int)
    
    # We use a set to store frames where a HARD CUT happened
    forced_cut_frames = set()

    prev_center = None 
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if DEBUG_MODE and frame_idx > (DEBUG_DURATION_LIMIT * fps): break

        cut_reason = ""
        is_teleport = False
        
        # --- INFERENCE ---
        results = model.predict(frame, conf=CONF_THRESHOLD, classes=[0, 32], imgsz=IMG_SIZE, verbose=False)
        ball_boxes = []
        person_boxes = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls == 0: person_boxes.append(box)
                elif cls == 32: ball_boxes.append(box)

        # --- PROCESS BALL ---
        valid_ball = False
        curr_center = None

        for box in ball_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w, h = x2 - x1, y2 - y1
            if w > 65 or h > 65: continue 
            
            # Shoe Filter is the ONLY safety we need
            if is_shoe(box, person_boxes):
                if DEBUG_MODE:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
                    cv2.putText(frame, "SHOE", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                continue 

            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # --- PHYSICS ---
            if prev_center is not None:
                _, speed = get_vector(prev_center, center)
                
                # SENSITIVE TELEPORT CHECK
                if speed > MAX_BALL_JUMP:
                    is_teleport = True
                    cut_reason = f"TELEPORT ({int(speed)}px)"
                
                if not is_teleport:
                    valid_ball = True
                    curr_center = center
            else:
                valid_ball = True
                curr_center = center
            
            if valid_ball or is_teleport: break 

        # --- REGISTER ---
        if is_teleport:
            forced_cut_frames.add(frame_idx)
            prev_center = None 
            if DEBUG_MODE: print(f"Frame {frame_idx}: ✂️ {cut_reason}")

        if valid_ball and not is_teleport:
            ball_timeline[frame_idx] = 1
            prev_center = curr_center
        else:
            prev_center = None

        if DEBUG_MODE:
            if is_teleport:
                cv2.rectangle(frame, (0,0), (w_frame, h_frame), (0, 0, 255), 20)
                cv2.putText(frame, cut_reason, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            if valid_ball:
                cx, cy = int(curr_center[0]), int(curr_center[1])
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            debug_writer.write(frame)
            if frame_idx % 10 == 0: print(f"Processing {frame_idx}...", end='\r')
        else:
            if frame_idx % 100 == 0: print(f"Scanning: {frame_idx}/{frames}", end='\r')
        
        frame_idx += 1

    cap.release()
    if debug_writer: debug_writer.release()
    
    if DEBUG_MODE:
        print("\n🛑 Debug Saved. Check 'debug_v18.mp4'")
        return

    # ================= SPLITTING =================
    print("\n\n📊 Slicing Timeline...")
    gap_frames = int(GAP_TOLERANCE_SEC * fps)
    
    rallies = []
    in_rally = False
    start_f = 0
    
    for i in range(len(ball_timeline)):
        
        # 1. HARD CUT (Teleport)
        if i in forced_cut_frames:
            if in_rally:
                in_rally = False
                end_f = i
                if (end_f - start_f) / fps > MIN_RALLY_DURATION:
                    rallies.append((start_f/fps, end_f/fps))
            continue 

        # 2. NORMAL FLOW
        if ball_timeline[i] == 1:
            if not in_rally:
                in_rally = True
                start_f = i
        else:
            if in_rally:
                # Look ahead for GAP_TOLERANCE
                # If we hit a FORCED CUT or run out of ball detections, we stop.
                stop_rally = True
                look_ahead = min(i + gap_frames, len(ball_timeline))
                
                for f in range(i, look_ahead):
                    if f in forced_cut_frames:
                        stop_rally = True
                        break
                    if ball_timeline[f] == 1:
                        stop_rally = False
                        break
                
                if stop_rally:
                    in_rally = False
                    end_f = i
                    if (end_f - start_f) / fps > MIN_RALLY_DURATION:
                        rallies.append((start_f/fps, end_f/fps))

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print(f"\n✂️ Final Count: {len(rallies)} rallies found.")
    for i, (start, end) in enumerate(rallies):
        out_name = os.path.join(OUTPUT_DIR, f"rally_{i+1:02d}.mp4")
        cmd = get_ffmpeg_cmd(VIDEO_PATH, start, end, out_name)
        subprocess.run(cmd)

    print("✅ Done.")

if __name__ == "__main__":
    detect_rallies()