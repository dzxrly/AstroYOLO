import os.path as osp

PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))
DATA_PATH = osp.join('./bhb/352x352/C0')

# train
TRAIN = {
    'TRAIN_IMG_SIZE': 352,
    'AUGMENT': False,
    'BATCH_SIZE': 16,
    'MULTI_SCALE_TRAIN': False,
    'IOU_THRESHOLD_LOSS': 0.5,
    'YOLO_EPOCHS': 120,
    'Mobilenet_YOLO_EPOCHS': 200,
    'NUMBER_WORKERS': 8,
    'MOMENTUM': 0.9,
    'WEIGHT_DECAY': 0.005,
    'LR_INIT': 1e-3,
    'LR_END': 1e-8,
    'WARMUP_EPOCHS': 50,  # or None
    'showatt': False
}

# val
VAL = {
    'EVAL_EPOCH': 30,
    'TEST_IMG_SIZE': 352,
    'BATCH_SIZE': 16,
    'NUMBER_WORKERS': 8,
    'CONF_THRESH': 0.005,
    'NMS_THRESH': 0.45,
    'MULTI_SCALE_VAL': False,
    'FLIP_VAL': False,
    'Visual': False,
    'showatt': False
}

Customer_DATA = {
    'NUM': 1,  # your dataset number
    'CLASSES': ['bhb'],  # your dataset class
}

# model
MODEL = {
    'ANCHORS': [
        [
            (1.25, 1.625),
            (2.0, 3.75),
            (4.125, 2.875),
        ],  # Anchors for small obj(12,16),(19,36),(40,28)
        [
            (1.875, 3.8125),
            (3.875, 2.8125),
            (3.6875, 7.4375),
        ],  # Anchors for medium obj(36,75),(76,55),(72,146)
        [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)],
    ],  # Anchors for big obj(142,110),(192,243),(459,401)
    'STRIDES': [8, 16, 32],
    'ANCHORS_PER_SCLAE': 3,
}
