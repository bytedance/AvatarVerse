import torch
import numpy as np
pose_shape = [1, 69]
valid_pose_num = 63
IRONMAN_STANDING_FIRE = torch.zeros(pose_shape)
IRONMAN_STANDING_FIRE[0, :valid_pose_num] = torch.tensor([0.0, 0.0, 0.0, -0.357010155916214, -0.3751673102378845, 0.17684337496757507, -0.27332618832588196, -0.09224412590265274, -0.41093185544013977, 0.0, -0.12654343247413635, 0.0, 0.4577910900115967, -0.07151380926370621, 0.08215107768774033, 0.2921690046787262, -0.20390288531780243, 0.3085102438926697, 0.0, 0.0, 0.0, -0.07447061687707901, -0.014327136799693108, -0.30494531989097595, 0.055549684911966324, -0.4499472379684448, 0.07973260432481766, -0.0009480331791564822, 0.001100416062399745, 0.0027252635918557644, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.06557828933000565, -0.9823535680770874, 0.06281368434429169, 0.0, -0.06447441130876541, 0.0, 0.0, 0.2247605323791504, 0.0, 0.0, 0.0, 0.0, 0.02809596247971058, -0.047661274671554565, -1.0650582313537598, -0.10616273432970047, 0.00019162155513186008, 0.04534279555082321, -0.2924211621284485, 0.09580747038125992, -0.37477126717567444, 0.0, 0.0, 0.0, 0.0, 0.0, -0.28281840682029724, -0.2201845794916153, 0.09501525014638901, -1.6506297588348389, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.4838465452194214, 0.0, 0.0, -1.5895321369171143, 0.0, 0.0, -0.9506988525390625, -0.154713973402977, -0.1764194816350937, -1.7404567003250122, -0.25381574034690857, 0.24664810299873352, -1.5390875339508057, 0.050932638347148895, 0.22550463676452637, -1.9439693689346313, -0.6216062903404236, -0.786018967628479, -1.6391586065292358, -0.836167573928833, -0.13927721977233887, -1.0118223428726196, -0.9470933079719543, -0.15775355696678162, -1.146050214767456, -0.11723831295967102, -0.6040090322494507, -0.9740037322044373, -0.579037070274353, -0.38831332325935364, -1.6968460083007812, 0.0, 0.0, -2.0743160247802734, -0.015098758041858673, 0.07890155911445618, -1.4917048929419252e-08, 1.6751983165740967, 0.040761273354291916, -0.5345197319984436, 0.33461207151412964, 1.297072410583496, -1.2688841819763184, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][3:3+valid_pose_num])

IRONMAN_STANDING_FIRE_FOCUS = [
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0., 0], dtype=torch.float), 0.4),  # close standard
                (torch.tensor([0., -0.45, 0], dtype=torch.float), 0.3),  # head
                (torch.tensor([0., 0.4, 0], dtype=torch.float), 0.4),  # foot
                # hand focus
                # (torch.tensor([0.4, -0.3, 0], dtype=torch.float), 0.3),  # close left hand
                (torch.tensor([-0.4, -0.3, 0], dtype=torch.float), 0.3),  # close right hand
            ]

'''
black widow
'''
BLACK_WIDOW_STANDING_FIRE = torch.zeros(pose_shape)
BLACK_WIDOW_STANDING_FIRE[0, :] = torch.tensor([-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.122999966, 1.882071e-09, 7.820344e-09, 0.5309357, -0.014106064, -0.0056014434, -0.0, -0.0, -0.0, -0.09299998, 4.653256e-10, 2.1653243e-11, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.07112146, -0.013427893, 0.20669392, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.037274078, -1.1488239, -0.024128843, -0.0, -0.0, -1.2900012, -0.0, -0.0, -0.0, -0.0, -0.0, -0.03000007, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.5867463, 0.02837321, -0.0044931625, -0.0, -0.0, -0.0, -0.03659925, -0.0261826, 0.045479387])

BLACK_WIDOW_STANDING_FIRE_FOCUS = [
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0., 0], dtype=torch.float), 0.4),  # close standard
                (torch.tensor([0., -0.4, 0], dtype=torch.float), 0.3),  # head
                (torch.tensor([0., -0.45, 0], dtype=torch.float), 0.4),  # head
                (torch.tensor([0., 0.4, 0], dtype=torch.float), 0.4),  # foot
                # hand focus
                # (torch.tensor([0.4, -0.3, 0], dtype=torch.float), 0.3),  # close left hand
                (torch.tensor([-0.4, -0.3, 0], dtype=torch.float), 0.3),  # close right hand
            ]


'''
Thur hammper
'''
THOR_STANDING = torch.zeros(pose_shape)
THOR_STANDING[0, :] = torch.tensor([-0.0, -0.0, 0.09000001, 1.2562584e-08, 6.223565e-10, -0.09899998, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -7.187477e-10, -0.44100013, -3.2066296e-09, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.13461442, 0.16397038, -0.093403146, -0.0, -0.0, -0.4919998, -0.83012414, 0.29131258, -0.63185424, -0.0, -0.0, -1.6020017, -0.42878112, 0.016730946, -0.0767998, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0
])

THOR_STANDING_FOCUS = [
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0., 0], dtype=torch.float), 0.4),  # close standard
                (torch.tensor([0., -0.45, 0], dtype=torch.float), 0.3),  # head
                (torch.tensor([0., 0.4, 0], dtype=torch.float), 0.4),  # foot
                # hand focus
                # (torch.tensor([0.4, -0.3, 0], dtype=torch.float), 0.3),  # close left hand
                (torch.tensor([-0.35, -0.45, 0], dtype=torch.float), 0.3),  # close right hand
            ]
'''
GENERAL  STANDING POSE
'''
GENERAL_STANDING = torch.zeros(pose_shape)
# left shoulder
GENERAL_STANDING[:, 38] = -np.pi / 6
GENERAL_STANDING[:, 47] = -np.pi / 4
# right shoulder
GENERAL_STANDING[:, 41] = np.pi / 6
GENERAL_STANDING[:, 50] = np.pi / 4

# GENERAL_STANDING[0, 2] = np.pi / 40
# # right shoulder
# GENERAL_STANDING[0, 5] = -np.pi / 40
GENERAL_STANDING_FOCUS = [
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0., 0], dtype=torch.float), 0.4),  # close standard
                (torch.tensor([0., -0.45, 0], dtype=torch.float), 0.3),  # head
                (torch.tensor([0., 0.4, 0], dtype=torch.float), 0.4),  # foot

            ]
'''
ARM OPEN  POSE
'''
ARM_OPEN_STANDING = torch.zeros(pose_shape)

ARM_OPEN_STANDING[0, 2] = np.pi / 40
# right shoulder
ARM_OPEN_STANDING[0, 5] = -np.pi / 40
ARM_OPEN_STANDING_FOCUS = [
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0., 0], dtype=torch.float), 0.4),  # close standard
                (torch.tensor([0., -0.45, 0], dtype=torch.float), 0.3),  # head
                (torch.tensor([0., 0.4, 0], dtype=torch.float), 0.4),  # foot

                (torch.tensor([0.4, -0.3, 0], dtype=torch.float), 0.3),  # close left hand
                (torch.tensor([-0.4, -0.3, 0], dtype=torch.float), 0.3),  # close right hand

            ]
'''
BIG POSE
'''
# big pose
GENERAL_BIG = torch.zeros(pose_shape)
GENERAL_BIG[0, 2] = np.pi / 8
# right shoulder
GENERAL_BIG[0, 5] = -np.pi / 8
GENERAL_BIG_FOCUS = [
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0., 0], dtype=torch.float), 0.4),  # close standard
                (torch.tensor([0., -0.45, 0], dtype=torch.float), 0.3),  # head
                (torch.tensor([0., 0.4, 0], dtype=torch.float), 0.4),  # foot
                # big pose, hand focus
                (torch.tensor([0.4, -0.3, 0], dtype=torch.float), 0.3),  # close left hand
                (torch.tensor([-0.4, -0.3, 0], dtype=torch.float), 0.3),  # close right hand
            ]


ONLY_STANDING_FOCUS = [
                (torch.tensor([0., -0.6, 0], dtype=torch.float), 0.3),  # head
                (torch.tensor([0., 0.7, 0], dtype=torch.float), 0.4),  # foot
    (torch.tensor([0.27, 0, 0], dtype=torch.float), 0.2),  # close left hand
    (torch.tensor([-0.27, 0, 0], dtype=torch.float), 0.2),  # close right hand

            ]


GENERAL_NO_FOCUS = [
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
            ]

GENERAL_SHOULDER_FOCUS = [
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., -0.45, 0], dtype=torch.float), 1.0),  # head
                (torch.tensor([0., -0.45, 0], dtype=torch.float), 1.0),  # head
                (torch.tensor([0., 0.4, 0], dtype=torch.float), 1.0),  # foot
            ]

HANDS_ON_HIPS = torch.tensor( [[0., 0., 0.05699999, 0., 0.,
          -0.057, 0., 0., 0., 0.,
          0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0.,
          0., 0., 0., 0.08399998, 0.,
          0., -0.09000004, 0., 0., 0.,
          -0.07566634, -0.25264037, -0.7218966, -0.08078638, 0.22586259,
          0.7674071, 0., 0., -1.6530018, 0.,
          0., 1.5030022, 0., 0., 0.7110004,
          0., 0., -0.5790002, 0., 0.,
          0., 0., 0., 0.]], dtype=torch.float32)
HANDS_ON_HIPS_FOCUS = [
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0, 0], dtype=torch.float), 1.0),  # closed standard
                (torch.tensor([0., -0.5, 0], dtype=torch.float), 0.6),  # head
                (torch.tensor([0., -0.6, 0], dtype=torch.float), 0.7),  # head
                (torch.tensor([0., 0.4, 0], dtype=torch.float), 0.5),  # foot
            ]

WALKING = torch.tensor([[-4.4399932e-01, -2.9505664e-03, -6.6600315e-04, 2.3699962e-01,
          -2.9859475e-03, 3.5550605e-04, 0.0000000e+00, 0.0000000e+00,
          0.0000000e+00, 2.3400010e-01, 7.6076745e-09, -3.2438736e-09,
          0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
          0.0000000e+00, 0.0000000e+00, 9.2999980e-02, 0.0000000e+00,
          0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
          0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
          0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
          0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
          0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
          0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
          0.0000000e+00, 1.4434952e-01, 1.8363307e-01, -1.4377967e+00,
          -3.6418009e-01, 5.1719791e-01, 1.3758763e+00, 0.0000000e+00,
          0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
          0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
          0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
          0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
          0.0000000e+00]], dtype=torch.float32)

WALKING_FOCUS = [
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0., 0], dtype=torch.float), 1.0),  # standard
                (torch.tensor([0., 0, 0], dtype=torch.float), 0.7),  # closed standard
                (torch.tensor([0., -0.5, 0], dtype=torch.float), 0.5),  # head
                (torch.tensor([0., 0.4, 0], dtype=torch.float), 0.4),  # foot

                (torch.tensor([0.1, 0., -0.1], dtype=torch.float), 0.4),  # close left hand
                (torch.tensor([-0.1, 0., 0.1], dtype=torch.float), 0.4),  # close right hand
            ]
'''
     0: 'pelvis',
     1: 'left_hip',
     2: 'right_hip',
     3: 'spine1',
     4: 'left_knee',
     5: 'right_knee',
     6: 'spine2',
     7: 'left_ankle',
     8: 'right_ankle',
     9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'left_hand',
    23: 'right_hand'
'''




