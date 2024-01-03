
# 145 joint, only 29 list for corresponding with openpose
SMPLX_JOINT_NAMES = [
        "OP_MidHip",
        "OP_LHip",
        "OP_RHip",
        "SMPL_SPIN1",
        "OP_LKnee",
        "OP_RKnee",
        "SMPL_SPIN2",
        "OP_LAnkle",
        "OP_RAnkle",
        "SMPL_SPIN3",
        "SMPL_Left_Foot",
        "SMPL_Right_Foot",
        "OP_Neck",
        "SMPL_Left_Collar",
        "SMPL_Right_Collar",
        "SMPL_Head",
        "OP_LShoulder",
        "OP_RShoulder",
        "OP_LElbow",
        "OP_RElbow",
        "OP_LWrist",
        "OP_RWrist",
        "jaw",
        "OP_LEye_backup",
        "OP_REye_backup",
        "left_index1",
        "left_index2",
        "left_index3",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",
        "right_index1",
        "right_index2",
        "right_index3",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3",
        "OP_Nose",
        "OP_REye",
        "OP_LEye",
        "OP_REar",
        "OP_LEar",
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
        "left_thumb",
        "left_index",
        "left_middle",
        "left_ring",
        "left_pinky",
        "right_thumb",
        "right_index",
        "right_middle",
        "right_ring",
        "right_pinky",
        "right_eye_brow1",
        "right_eye_brow2",
        "right_eye_brow3",
        "right_eye_brow4",
        "right_eye_brow5",
        "left_eye_brow5",
        "left_eye_brow4",
        "left_eye_brow3",
        "left_eye_brow2",
        "left_eye_brow1",
        "nose1",
        "nose2",
        "nose3",
        "nose4",
        "right_nose_2",
        "right_nose_1",
        "nose_middle",
        "left_nose_1",
        "left_nose_2",
        "right_eye1",
        "right_eye2",
        "right_eye3",
        "right_eye4",
        "right_eye5",
        "right_eye6",
        "left_eye4",
        "left_eye3",
        "left_eye2",
        "left_eye1",
        "left_eye6",
        "left_eye5",
        "right_mouth_1",
        "right_mouth_2",
        "right_mouth_3",
        "mouth_top",
        "left_mouth_3",
        "left_mouth_2",
        "left_mouth_1",
        "left_mouth_5",
        "left_mouth_4",
        "mouth_bottom",
        "right_mouth_4",
        "right_mouth_5",
        "right_lip_1",
        "right_lip_2",
        "lip_top",
        "left_lip_2",
        "left_lip_1",
        "left_lip_3",
        "lip_bottom",
        "right_lip_3",
        "right_contour_1",
        "right_contour_2",
        "right_contour_3",
        "right_contour_4",
        "right_contour_5",
        "right_contour_6",
        "right_contour_7",
        "right_contour_8",
        "contour_middle",
        "left_contour_8",
        "left_contour_7",
        "left_contour_6",
        "left_contour_5",
        "left_contour_4",
        "left_contour_3",
        "left_contour_2",
        "left_contour_1"
]
SMPLX_JOINT_IDS = {SMPLX_JOINT_NAMES[i]: i for i in range(len(SMPLX_JOINT_NAMES))}

# 25 body joint
OPENPOSE25_JOINT_NAMES = [
        'OP_Nose', 'OP_Neck', 'OP_RShoulder',           #0,1,2
        'OP_RElbow', 'OP_RWrist', 'OP_LShoulder',       #3,4,5
        'OP_LElbow', 'OP_LWrist', 'OP_MidHip',          #6, 7,8
        'OP_RHip', 'OP_RKnee', 'OP_RAnkle',             #9,10,11
        'OP_LHip', 'OP_LKnee', 'OP_LAnkle',             #12,13,14
        'OP_REye', 'OP_LEye', 'OP_REar',                #15,16,17
        'OP_LEar', 'OP_LBigToe', 'OP_LSmallToe',        #18,19,20
        'OP_LHeel', 'OP_RBigToe', 'OP_RSmallToe', 'OP_RHeel',  #21, 22, 23, 24  ##Total 25 joints  for openpose
]
OPENPOSE25_JOINT_IDS = {OPENPOSE25_JOINT_NAMES[i]: i for i in range(len(OPENPOSE25_JOINT_NAMES))}

# 18 body joints
OPENPOSE18_JOINT_NAMES=[
        'OP_Nose', 'OP_Neck', 'OP_RShoulder',           #0,1,2
        'OP_RElbow', 'OP_RWrist', 'OP_LShoulder',       #3,4,5
        'OP_LElbow', 'OP_LWrist',                       #6, 7
        'OP_RHip', 'OP_RKnee', 'OP_RAnkle',             #8, 9,10,
        'OP_LHip', 'OP_LKnee', 'OP_LAnkle',             #11, 12,13,
        'OP_REye', 'OP_LEye', 'OP_REar',                #14, 15,16,
        'OP_LEar',                                      #17
]
OPENPOSE18_JOINT_IDS = {OPENPOSE18_JOINT_NAMES[i]: i for i in range(len(OPENPOSE18_JOINT_NAMES))}

SMPL_INDEX_FOR_OPENPOSE18 = [SMPLX_JOINT_IDS[name] for name in OPENPOSE18_JOINT_NAMES]