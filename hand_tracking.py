# ============================================================
# AIRCONTROL AI
# FINAL GESTURE CONTROLLER
#
# ☝️  INDEX ONLY          -> MOVE CURSOR
# 🤏  THUMB + INDEX       -> LEFT CLICK
# ✌️  TWO FINGERS UP      -> SCROLL UP
# ✌️  TWO FINGERS DOWN    -> SCROLL DOWN
# ✊  FIST                -> PAUSE
# 👍  THUMBS UP           -> RESUME
#
# PAUSED:
# ONLY THUMBS UP CAN RESUME
# ============================================================

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math
import time


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"D:\projects\AirControl-AI\models\hand_landmarker.task"

CAMERA_INDEX = 0

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


# ============================================================
# MOUSE
# ============================================================

MARGIN_X = 0.08
MARGIN_Y = 0.08

MOUSE_SMOOTHING = 0.78

SAFE_BORDER = 10


# ============================================================
# PINCH
# ============================================================

PINCH_START_DISTANCE = 0.085
PINCH_RELEASE_DISTANCE = 0.12

PINCH_CONFIRM_FRAMES = 2


# ============================================================
# FIST
# ============================================================

# IMPORTANT:
# Fist requires ALL FOUR fingers to be folded.
FIST_CONFIRM_FRAMES = 8

# Lower than this = folded.
FIST_FINGER_ANGLE = 145


# ============================================================
# THUMBS UP
# ============================================================

THUMBS_UP_CONFIRM_FRAMES = 8


# ============================================================
# SCROLL
# ============================================================

SCROLL_UP_AMOUNT = 15
SCROLL_DOWN_AMOUNT = -15


# ============================================================
# PYAUTOGUI
# ============================================================

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.55,
    min_hand_presence_confidence=0.55,
    min_tracking_confidence=0.55
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_MSMF
)

if not cap.isOpened():
    print("ERROR: Could not open Camo camera.")
    exit()


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# CURSOR SMOOTHER
# ============================================================

class CursorSmoother:

    def __init__(self):
        self.x = None
        self.y = None

    def reset(self):
        self.x = None
        self.y = None

    def update(self, x, y):

        if self.x is None:
            self.x = x
            self.y = y

            return self.x, self.y

        self.x += (
            x - self.x
        ) * MOUSE_SMOOTHING

        self.y += (
            y - self.y
        ) * MOUSE_SMOOTHING

        return self.x, self.y


cursor_smoother = CursorSmoother()


# ============================================================
# STATE
# ============================================================

paused = False

fist_counter = 0
thumb_counter = 0
pinch_counter = 0

pinch_active = False

mouse_x = SCREEN_WIDTH / 2
mouse_y = SCREEN_HEIGHT / 2

start_time = time.time()


# ============================================================
# MATH
# ============================================================

def distance_2d(a, b):

    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


def angle(a, b, c):

    ba = (
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    )

    bc = (
        c.x - b.x,
        c.y - b.y,
        c.z - b.z
    )

    dot = (
        ba[0] * bc[0] +
        ba[1] * bc[1] +
        ba[2] * bc[2]
    )

    mag_ba = math.sqrt(
        ba[0] ** 2 +
        ba[1] ** 2 +
        ba[2] ** 2
    )

    mag_bc = math.sqrt(
        bc[0] ** 2 +
        bc[1] ** 2 +
        bc[2] ** 2
    )

    if mag_ba == 0 or mag_bc == 0:
        return 0

    cosine = (
        dot /
        (mag_ba * mag_bc)
    )

    cosine = max(
        -1.0,
        min(1.0, cosine)
    )

    return math.degrees(
        math.acos(cosine)
    )


# ============================================================
# FINGER SHAPE
# ============================================================

def finger_angle(hand, mcp, pip, tip):

    return angle(
        hand[mcp],
        hand[pip],
        hand[tip]
    )


def finger_extended(hand, mcp, pip, tip):

    return (
        finger_angle(
            hand,
            mcp,
            pip,
            tip
        )
        > 145
    )


def finger_folded(hand, mcp, pip, tip):

    return (
        finger_angle(
            hand,
            mcp,
            pip,
            tip
        )
        < FIST_FINGER_ANGLE
    )


# ============================================================
# INDIVIDUAL FINGERS
# ============================================================

def index_up(hand):

    return finger_extended(
        hand,
        5,
        6,
        8
    )


def middle_up(hand):

    return finger_extended(
        hand,
        9,
        10,
        12
    )


def ring_up(hand):

    return finger_extended(
        hand,
        13,
        14,
        16
    )


def pinky_up(hand):

    return finger_extended(
        hand,
        17,
        18,
        20
    )


# ============================================================
# THUMB
# ============================================================

def thumb_up(hand):

    thumb_angle = angle(
        hand[2],
        hand[3],
        hand[4]
    )

    vertical = (
        hand[4].y < hand[3].y
        and
        hand[4].y < hand[2].y
    )

    return (
        thumb_angle > 140
        and
        vertical
    )


# ============================================================
# THUMBS UP
# ============================================================

def detect_thumbs_up(hand):

    if not thumb_up(hand):
        return False

    if index_up(hand):
        return False

    if middle_up(hand):
        return False

    if ring_up(hand):
        return False

    if pinky_up(hand):
        return False

    return True


# ============================================================
# PALM SIZE
# ============================================================

def palm_size(hand):

    wrist = hand[0]

    d1 = distance_2d(
        wrist,
        hand[5]
    )

    d2 = distance_2d(
        wrist,
        hand[9]
    )

    d3 = distance_2d(
        wrist,
        hand[17]
    )

    return (
        d1 + d2 + d3
    ) / 3


# ============================================================
# PINCH DISTANCE
# ============================================================

def get_pinch_distance(hand):

    return distance_2d(
        hand[4],
        hand[8]
    )


# ============================================================
# PINCH DETECTION
# ============================================================

def detect_pinch(hand):

    # Middle, ring and pinky MUST be folded.
    if middle_up(hand):
        return False

    if ring_up(hand):
        return False

    if pinky_up(hand):
        return False


    pinch_distance = get_pinch_distance(hand)


    # Thumb and index must be close.
    if pinch_distance > PINCH_START_DISTANCE:
        return False


    # --------------------------------------------------------
    # Make sure this is not simply a completely closed fist.
    # --------------------------------------------------------

    palm = palm_size(hand)

    if palm < 0.001:
        return False


    index_wrist_distance = distance_2d(
        hand[8],
        hand[0]
    )


    # In a proper pinch, index fingertip is still somewhat
    # forward from the wrist.
    if index_wrist_distance < palm * 1.10:
        return False


    return True


def pinch_released(hand):

    return (
        get_pinch_distance(hand)
        >
        PINCH_RELEASE_DISTANCE
    )


# ============================================================
# FIST DETECTION
# ============================================================

def detect_fist(hand):

    # --------------------------------------------------------
    # THUMBS UP IS NOT A FIST
    # --------------------------------------------------------

    if thumb_up(hand):
        return False


    # --------------------------------------------------------
    # CRITICAL:
    #
    # ALL FOUR fingers must be folded.
    #
    # This prevents normal INDEX movement from becoming
    # a fist.
    # --------------------------------------------------------

    index_folded = finger_folded(
        hand,
        5,
        6,
        8
    )

    middle_folded = finger_folded(
        hand,
        9,
        10,
        12
    )

    ring_folded = finger_folded(
        hand,
        13,
        14,
        16
    )

    pinky_folded = finger_folded(
        hand,
        17,
        18,
        20
    )


    # ALL FOUR required.
    if not (
        index_folded
        and
        middle_folded
        and
        ring_folded
        and
        pinky_folded
    ):
        return False


    # --------------------------------------------------------
    # FIST COMPACTNESS
    # --------------------------------------------------------

    palm = palm_size(hand)

    if palm < 0.001:
        return False


    wrist = hand[0]


    index_distance = distance_2d(
        hand[8],
        wrist
    )

    middle_distance = distance_2d(
        hand[12],
        wrist
    )

    ring_distance = distance_2d(
        hand[16],
        wrist
    )

    pinky_distance = distance_2d(
        hand[20],
        wrist
    )


    # Closed fist fingertips should be relatively close
    # to the wrist/palm.
    compact_limit = palm * 2.65


    compact_count = sum([
        index_distance < compact_limit,
        middle_distance < compact_limit,
        ring_distance < compact_limit,
        pinky_distance < compact_limit
    ])


    # At least 3 fingertips must be compact.
    if compact_count < 3:
        return False


    # --------------------------------------------------------
    # PINCH EXCLUSION
    # --------------------------------------------------------
    #
    # A real pinch has thumb + index extremely close while
    # the index fingertip remains forward.
    #
    # Do NOT reject every small pinch distance because that
    # would make fists fail again.
    # --------------------------------------------------------

    pinch_distance = get_pinch_distance(hand)

    if (
        pinch_distance < 0.055
        and
        index_distance > palm * 1.10
    ):
        return False


    return True


# ============================================================
# SCROLL
# ============================================================

def detect_scroll(hand):

    if not index_up(hand):
        return None

    if not middle_up(hand):
        return None

    if ring_up(hand):
        return None

    if pinky_up(hand):
        return None


    index_direction = (
        hand[8].y -
        hand[5].y
    )

    middle_direction = (
        hand[12].y -
        hand[9].y
    )


    # Two fingers pointing UP.
    if (
        index_direction < -0.06
        and
        middle_direction < -0.06
    ):
        return "UP"


    # Two fingers pointing DOWN.
    if (
        index_direction > 0.06
        and
        middle_direction > 0.06
    ):
        return "DOWN"


    return None


# ============================================================
# MOUSE
# ============================================================

def move_mouse(hand):

    global mouse_x
    global mouse_y


    raw_x = hand[8].x
    raw_y = hand[8].y


    raw_x = max(
        MARGIN_X,
        min(
            1.0 - MARGIN_X,
            raw_x
        )
    )

    raw_y = max(
        MARGIN_Y,
        min(
            1.0 - MARGIN_Y,
            raw_y
        )
    )


    normalized_x = (
        raw_x -
        MARGIN_X
    ) / (
        1.0 -
        2 * MARGIN_X
    )


    normalized_y = (
        raw_y -
        MARGIN_Y
    ) / (
        1.0 -
        2 * MARGIN_Y
    )


    target_x = (
        normalized_x *
        SCREEN_WIDTH
    )

    target_y = (
        normalized_y *
        SCREEN_HEIGHT
    )


    filtered_x, filtered_y = (
        cursor_smoother.update(
            target_x,
            target_y
        )
    )


    dx = filtered_x - mouse_x
    dy = filtered_y - mouse_y


    movement = math.sqrt(
        dx ** 2 +
        dy ** 2
    )


    MAX_STEP = 150


    if movement > MAX_STEP:

        scale = (
            MAX_STEP /
            movement
        )

        dx *= scale
        dy *= scale


    mouse_x += dx
    mouse_y += dy


    mouse_x = max(
        SAFE_BORDER,
        min(
            SCREEN_WIDTH -
            SAFE_BORDER,
            mouse_x
        )
    )


    mouse_y = max(
        SAFE_BORDER,
        min(
            SCREEN_HEIGHT -
            SAFE_BORDER,
            mouse_y
        )
    )


    pyautogui.moveTo(
        int(mouse_x),
        int(mouse_y),
        _pause=False
    )


# ============================================================
# SCROLL ACTION
# ============================================================

def perform_scroll(direction):

    if direction == "UP":

        pyautogui.scroll(
            SCROLL_UP_AMOUNT
        )

        return "⬆️ SCROLL UP"


    if direction == "DOWN":

        pyautogui.scroll(
            SCROLL_DOWN_AMOUNT
        )

        return "⬇️ SCROLL DOWN"


    return "READY"


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(frame, hand):

    height, width, _ = frame.shape


    connections = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (0, 17)
    ]


    for start, end in connections:

        x1 = int(
            hand[start].x *
            width
        )

        y1 = int(
            hand[start].y *
            height
        )

        x2 = int(
            hand[end].x *
            width
        )

        y2 = int(
            hand[end].y *
            height
        )


        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )


    for point in hand:

        x = int(
            point.x *
            width
        )

        y = int(
            point.y *
            height
        )


        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )


# ============================================================
# CAMERA WINDOW
# ============================================================

WINDOW_NAME = "AirControl AI"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
)

cv2.resizeWindow(
    WINDOW_NAME,
    960,
    540
)

cv2.moveWindow(
    WINDOW_NAME,
    100,
    100
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print(
            "ERROR: Could not read camera."
        )

        break


    # --------------------------------------------------------
    # NO MIRROR
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp_ms = int(
        (
            time.time()
            -
            start_time
        )
        *
        1000
    )


    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    gesture_text = "NO HAND"


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]


        draw_hand(
            frame,
            hand
        )


        # ====================================================
        # PAUSED
        # ====================================================

        if paused:

            pinch_active = False
            pinch_counter = 0
            fist_counter = 0


            if detect_thumbs_up(hand):

                thumb_counter += 1


                if (
                    thumb_counter
                    >=
                    THUMBS_UP_CONFIRM_FRAMES
                ):

                    paused = False

                    thumb_counter = 0

                    cursor_smoother.reset()

                    mouse_x = (
                        SCREEN_WIDTH / 2
                    )

                    mouse_y = (
                        SCREEN_HEIGHT / 2
                    )


                    gesture_text = (
                        "👍 RESUMED"
                    )

                else:

                    gesture_text = (
                        "👍 HOLD TO RESUME"
                    )


            else:

                thumb_counter = 0

                gesture_text = (
                    "⏸️ PAUSED"
                )


        # ====================================================
        # ACTIVE
        # ====================================================

        else:

            # =================================================
            # FIST FIRST
            # =================================================

            if detect_fist(hand):

                fist_counter += 1

                pinch_counter = 0
                pinch_active = False


                if (
                    fist_counter
                    >=
                    FIST_CONFIRM_FRAMES
                ):

                    paused = True

                    fist_counter = 0

                    gesture_text = (
                        "✊ PAUSED"
                    )

                else:

                    gesture_text = (
                        "✊ HOLD TO PAUSE"
                    )


            else:

                # IMPORTANT:
                # Reset immediately when hand isn't a fist.
                fist_counter = 0


                # =============================================
                # PINCH
                # =============================================

                if pinch_active:

                    if pinch_released(hand):

                        pinch_active = False

                        pinch_counter = 0

                        gesture_text = (
                            "READY"
                        )

                    else:

                        gesture_text = (
                            "🤏 PINCH"
                        )


                elif detect_pinch(hand):

                    pinch_counter += 1


                    if (
                        pinch_counter
                        >=
                        PINCH_CONFIRM_FRAMES
                    ):

                        pinch_active = True

                        pinch_counter = 0

                        pyautogui.click()

                        gesture_text = (
                            "🤏 LEFT CLICK"
                        )

                    else:

                        gesture_text = (
                            "🤏 PINCH..."
                        )


                else:

                    pinch_counter = 0


                    # =========================================
                    # THUMBS UP
                    # =========================================

                    if detect_thumbs_up(hand):

                        thumb_counter += 1


                        if (
                            thumb_counter
                            >=
                            THUMBS_UP_CONFIRM_FRAMES
                        ):

                            gesture_text = (
                                "👍 ACTIVE"
                            )

                        else:

                            gesture_text = (
                                "👍 HOLD..."
                            )


                    else:

                        thumb_counter = 0


                        # =====================================
                        # SCROLL
                        # =====================================

                        scroll_direction = (
                            detect_scroll(hand)
                        )


                        if (
                            scroll_direction
                            ==
                            "UP"
                        ):

                            gesture_text = (
                                perform_scroll(
                                    "UP"
                                )
                            )


                        elif (
                            scroll_direction
                            ==
                            "DOWN"
                        ):

                            gesture_text = (
                                perform_scroll(
                                    "DOWN"
                                )
                            )


                        # =====================================
                        # INDEX MOVE
                        # =====================================

                        elif index_up(hand):

                            if (
                                not middle_up(hand)
                                and
                                not ring_up(hand)
                                and
                                not pinky_up(hand)
                            ):

                                move_mouse(hand)

                                gesture_text = (
                                    "☝️ MOVE"
                                )

                            else:

                                gesture_text = (
                                    "READY"
                                )


                        else:

                            gesture_text = (
                                "READY"
                            )


    # ========================================================
    # NO HAND
    # ========================================================

    else:

        fist_counter = 0
        thumb_counter = 0
        pinch_counter = 0
        pinch_active = False

        gesture_text = (
            "NO HAND"
        )


    # ========================================================
    # DEBUG
    # ========================================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]


        pinch_distance = (
            get_pinch_distance(hand)
        )

        index_angle = finger_angle(
            hand,
            5,
            6,
            8
        )

        middle_angle = finger_angle(
            hand,
            9,
            10,
            12
        )

        ring_angle = finger_angle(
            hand,
            13,
            14,
            16
        )

        pinky_angle = finger_angle(
            hand,
            17,
            18,
            20
        )


        debug_1 = (
            f"Pinch: {pinch_distance:.3f}"
            f" | Fist count: {fist_counter}/{FIST_CONFIRM_FRAMES}"
        )


        debug_2 = (
            f"Angles: "
            f"I {index_angle:.0f}"
            f" M {middle_angle:.0f}"
            f" R {ring_angle:.0f}"
            f" P {pinky_angle:.0f}"
        )


        cv2.putText(
            frame,
            debug_1,
            (30, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            debug_2,
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2
        )


    # ========================================================
    # STATUS
    # ========================================================

    if paused:

        status = (
            "STATUS: PAUSED | "
            "ONLY 👍 THUMBS UP CAN RESUME"
        )

    else:

        status = (
            "STATUS: ACTIVE"
        )


    # ========================================================
    # UI
    # ========================================================

    cv2.putText(
        frame,
        gesture_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (0, 255, 0),
        3
    )


    cv2.putText(
        frame,
        status,
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "INDEX = MOVE | PINCH = CLICK | 2 FINGERS = SCROLL",
        (30, frame.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "FIST = PAUSE | THUMBS UP = RESUME | Q = QUIT",
        (30, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

landmarker.close()

cv2.destroyAllWindows()

print(
    "AirControl AI stopped successfully."
)