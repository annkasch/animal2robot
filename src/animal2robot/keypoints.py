KEYPOINT_NAMES: list[str] = [
    "nose",                  # 0
    "left_eye",              # 1
    "right_eye",             # 2
    "left_ear",              # 3
    "right_ear",             # 4
    "throat",                # 5
    "withers",               # 6
    "tail_base",             # 7
    "left_front_paw",        # 8
    "left_front_wrist",      # 9
    "left_front_elbow",      # 10
    "left_front_shoulder",   # 11
    "right_front_paw",       # 12
    "right_front_wrist",     # 13
    "right_front_elbow",     # 14
    "right_front_shoulder",  # 15
    "left_back_paw",         # 16
    "left_back_hock",        # 17
    "left_back_knee",        # 18
    "left_back_hip",         # 19
    "right_back_paw",        # 20
    "right_back_hock",       # 21
    "right_back_knee",       # 22
    "right_back_hip",        # 23
]

PAW_INDICES: list[int] = [8, 12, 16, 20]

PAW_NAMES: list[str] = [KEYPOINT_NAMES[i] for i in PAW_INDICES]

DOG_SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 5),   # nose → throat
    (5, 6),   # throat → withers
    (6, 7),   # withers → tail_base
    # Front left leg
    (6, 11), (11, 10), (10, 9), (9, 8),
    # Front right leg
    (6, 15), (15, 14), (14, 13), (13, 12),
    # Rear left leg
    (7, 19), (19, 18), (18, 17), (17, 16),
    # Rear right leg
    (7, 23), (23, 22), (22, 21), (21, 20),
]
