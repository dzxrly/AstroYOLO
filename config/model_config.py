import os.path as osp

PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))
DATA_PATH = osp.join('dataset_example/dataset')  # your dataset path (absolute path is recommended)

# train
TRAIN = {
    'amp': True,  # use amp or not
    'TRAIN_IMG_SIZE': 352,  # training image size
    'BATCH_SIZE': 16,  # training batch size
    'IOU_THRESHOLD_LOSS': 0.5,  # iou threshold for loss
    'YOLO_EPOCHS': 120,  # total training epochs
    'NUMBER_WORKERS': 8,  # number of workers for dataloader
    'MOMENTUM': 0.9,  # SGD momentum
    'WEIGHT_DECAY': 0.005,  # SGD weight decay
    'LR_INIT': 1e-3,  # initial learning rate
    'LR_END': 1e-8,  # end learning rate
    'WARMUP_EPOCHS': 50,  # warmup epochs
}

# val
VAL = {
    'EVAL_EPOCH': 30,  # evaluation begin epoch
    'TEST_IMG_SIZE': 352,  # test image size, be same as training image size!
    'BATCH_SIZE': 16,  # test batch size, be same as training batch size!
    'NUMBER_WORKERS': 8,  # number of workers for dataloader
    'CONF_THRESH': 0.005,  # confidence threshold for evaluation
    'NMS_THRESH': 0.45,  # nms threshold for evaluation
    'MULTI_SCALE_VAL': False,
    'FLIP_VAL': False,
}

Customer_DATA = {
    'NUM': 1,  # your dataset number
    'CLASSES': ['bhb'],  # your dataset class
}

# model
MODEL = {
    'ANCHORS': [  # anchors for three scale, be careful to change it!!!
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
