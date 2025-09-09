import cv2
import numpy as np
from pupil_apriltags import Detector

# ---- Camera parameters (replace with calibration values for better accuracy) ----
fx, fy = 600, 600   # focal length (pixels)
cx, cy = 320, 240   # optical center (image center for 640x480)

camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float32)

dist_coeffs = np.zeros(5)  # assuming no lens distortion

# ---- AprilTag Detector ----
at_detector = Detector(families="tag36h11")

cap = cv2.VideoCapture(0)

tag_size = 0.05  # size of tag in meters (5 cm)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = at_detector.detect(gray, estimate_tag_pose=False)

    for det in detections:
        corners = np.array(det.corners, dtype=np.float32)

        # Define 3D object points of tag corners
        obj_points = np.array([
            [-tag_size/2,  tag_size/2, 0],
            [ tag_size/2,  tag_size/2, 0],
            [ tag_size/2, -tag_size/2, 0],
            [-tag_size/2, -tag_size/2, 0]
        ], dtype=np.float32)

        # Solve PnP (pose estimation)
        success, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)
        if not success:
            continue

        # ---- Apply coordinate frame rotation fix ----
        R, _ = cv2.Rodrigues(rvec)
        R_fix = np.array([[1, 0, 0],
                          [0,-1, 0],
                          [0, 0,-1]], dtype=np.float32)
        R = R @ R_fix
        rvec, _ = cv2.Rodrigues(R)

        # Draw bounding box
        for i in range(4):
            pt1 = tuple(corners[i].astype(int))
            pt2 = tuple(corners[(i+1)%4].astype(int))
            cv2.line(frame, pt1, pt2, (0,255,0), 2)

        # ---- Draw 3D axes from the center of the tag ----
        axis_length = 0.05  # 5 cm
        axis_points = np.float32([
            [axis_length,0,0],  # X axis (red)
            [0,axis_length,0],  # Y axis (green)
            [0,0,axis_length]   # Z axis (blue)
        ]).reshape(-1,3)

        # Center of the tag
        tag_center = np.mean(corners, axis=0).astype(int)

        imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

        # Draw axes from center
        cv2.line(frame, tuple(tag_center), tuple(imgpts[0].ravel().astype(int)), (0,0,255), 3) # X red
        cv2.line(frame, tuple(tag_center), tuple(imgpts[1].ravel().astype(int)), (0,255,0), 3) # Y green
        cv2.line(frame, tuple(tag_center), tuple(imgpts[2].ravel().astype(int)), (255,0,0), 3) # Z blue

        # ---- Euler angles (roll, pitch, yaw) ----
        sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])

        # ---- Display info in black text ----
        cv2.putText(frame, f"ID:{det.tag_id} X:{tvec[0][0]:.2f}m Y:{tvec[1][0]:.2f}m Z:{tvec[2][0]:.2f}m",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(frame, f"Roll:{roll:.2f} Pitch:{pitch:.2f} Yaw:{yaw:.2f}",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    cv2.imshow("AprilTag Pose", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
