import cv2
import mediapipe as mp
import pyautogui
import time
import math

#  Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

screen_w, screen_h = pyautogui.size()
print("Hand Mouse Control Started")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not opening")
    exit()

cv2.namedWindow("Live Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Video", 800, 600)

#  Click 
click_times = []
click_cooldown = 0.6

#  Movement smoothing 
prev_x, prev_y = 0, 0
smoothing = 5  # bigger = smoother but slower

last_move_time = 0
move_delay = 0.02

#  Scroll 
prev_wrist_y = None
scroll_delay = 0.15
last_scroll_time = 0
scroll_threshold = 0.015   # hand movement sensitivity

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            #  Landmarks
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            wrist = hand_landmarks.landmark[0]

            current_time = time.time()

            # CLICK 
            dist = math.hypot(
                thumb_tip.x - index_tip.x,
                thumb_tip.y - index_tip.y
            )

            if dist < 0.05:
                if len(click_times) == 0 or current_time - click_times[-1] > click_cooldown:
                    click_times.append(current_time)

                    if len(click_times) >= 2 and click_times[-1] - click_times[-2] < 0.4:
                        # DOUBLE CLICK SAFE
                        pyautogui.click()  # 1st click
                        time.sleep(0.05)
                        pyautogui.click()  # 2nd click
                        cv2.putText(frame, "Double Click", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        click_times = []
                    else:
                        pyautogui.click()
                        cv2.putText(frame, "Single Click", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            #  SMOOTH MOUSE MOVE
            if current_time - last_move_time > move_delay:
                target_x = int(index_tip.x * screen_w)
                target_y = int(index_tip.y * screen_h)

                # interpolate
                smooth_x = prev_x + (target_x - prev_x) / smoothing
                smooth_y = prev_y + (target_y - prev_y) / smoothing

                pyautogui.moveTo(smooth_x, smooth_y)
                prev_x, prev_y = smooth_x, smooth_y
                last_move_time = current_time

            # PALM SCROLL (UP / DOWN) 
            if prev_wrist_y is not None and current_time - last_scroll_time > scroll_delay:
                diff = wrist.y - prev_wrist_y

                if diff > scroll_threshold:
                    pyautogui.scroll(-300)   # scroll down
                    cv2.putText(frame, "Scroll Down", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    last_scroll_time = current_time

                elif diff < -scroll_threshold:
                    pyautogui.scroll(300)    # scroll up
                    cv2.putText(frame, "Scroll Up", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    last_scroll_time = current_time

            prev_wrist_y = wrist.y

    cv2.imshow("Live Video", frame)

    #  Exit 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
