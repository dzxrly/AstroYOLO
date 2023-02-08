import argparse

import torch.cuda as cuda
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import config.model_config as cfg
import utils.datasets as data
import utils.gpu as gpu
from eval.evaluator import *
from model.loss.yolo_loss import YOLOLoss
from model.model import BuildModel
from utils import cosine_lr_scheduler
from utils.fits_operator import create_dir
from utils.tools import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Trainer(object):
    def __init__(self, gpu_id=0, fp_16=False):
        super(Trainer, self).__init__()
        init_seeds(0)
        self.fp_16 = fp_16
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.0
        self.weight_path = ''
        self.train_dataset = data.BuildDataset(anno_file_type='train', img_size=cfg.TRAIN['TRAIN_IMG_SIZE'])
        self.classes_num = self.train_dataset.num_classes - 1 if 'unknown' in cfg.Customer_DATA[
            'CLASSES'] else self.train_dataset.num_classes
        self.epochs = cfg.TRAIN['YOLO_EPOCHS']
        self.eval_epoch = cfg.VAL['EVAL_EPOCH']
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=cfg.TRAIN['BATCH_SIZE'],
            num_workers=cfg.TRAIN['NUMBER_WORKERS'],
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        self.yolov4 = BuildModel().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.yolov4.parameters(),
            lr=cfg.TRAIN['LR_INIT'],
            momentum=cfg.TRAIN['MOMENTUM'],
            weight_decay=cfg.TRAIN['WEIGHT_DECAY'],
        )
        self.criterion = YOLOLoss(
            anchors=cfg.MODEL['ANCHORS'],
            strides=cfg.MODEL['STRIDES'],
            iou_threshold_loss=cfg.TRAIN['IOU_THRESHOLD_LOSS'],
        )
        self.scheduler = cosine_lr_scheduler.CosineDecayLR(
            self.optimizer,
            T_max=self.epochs * len(self.train_dataloader),
            lr_init=cfg.TRAIN['LR_INIT'],
            lr_min=cfg.TRAIN['LR_END'],
            warmup=cfg.TRAIN['WARMUP_EPOCHS'] * len(self.train_dataloader),
        )

    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(self.weight_path, 'best.pt')
        last_weight = os.path.join(self.weight_path, 'last.pt')
        chkpt = {
            'epoch': epoch,
            'best_mAP': self.best_mAP,
            'model': self.yolov4.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(chkpt, last_weight)
        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)
        if epoch > 0 and epoch % 10 == 0:
            torch.save(chkpt, os.path.join(self.weight_path, 'backup_epoch%g.pt' % epoch))
        del chkpt

    def train(self):
        start_time = time.time()
        writer = SummaryWriter(
            logdir='./logs/{}-Epoch_{}-Batch_{}'.format(start_time, self.epochs, cfg.TRAIN['BATCH_SIZE']))
        create_dir('./files/{}-Epoch_{}-Batch_{}'.format(start_time, self.epochs, cfg.TRAIN['BATCH_SIZE']))
        self.weight_path = './files/{}-Epoch_{}-Batch_{}'.format(start_time, self.epochs, cfg.TRAIN['BATCH_SIZE'])

        def is_valid_number(x):
            return not (math.isnan(x) or math.isinf(x) or x > 1e4)

        grad_scaler = None
        if self.fp_16:
            print('Using torch.amp fp16 training.')
            grad_scaler = cuda.amp.GradScaler(enabled=True)
        for epoch in range(self.start_epoch, self.epochs):
            self.yolov4.train()
            mloss = torch.zeros(4)
            # Training
            with tqdm(total=len(self.train_dataloader), unit='batch', ncols=120,
                      desc='[{}/{} Epoch] Training'.format(epoch + 1, self.epochs)) as pbar:
                for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(
                        self.train_dataloader):
                    self.scheduler.step(len(self.train_dataloader) / (cfg.TRAIN['BATCH_SIZE']) * epoch + i)
                    imgs = imgs.to(self.device)
                    label_sbbox = label_sbbox.to(self.device)
                    label_mbbox = label_mbbox.to(self.device)
                    label_lbbox = label_lbbox.to(self.device)
                    sbboxes = sbboxes.to(self.device)
                    mbboxes = mbboxes.to(self.device)
                    lbboxes = lbboxes.to(self.device)
                    with cuda.amp.autocast(enabled=self.fp_16):
                        p, p_d = self.yolov4(imgs)
                        loss, loss_ciou, loss_conf, loss_cls = self.criterion(
                            p,
                            p_d,
                            label_sbbox,
                            label_mbbox,
                            label_lbbox,
                            sbboxes,
                            mbboxes,
                            lbboxes,
                        )
                    if is_valid_number(loss.item()):
                        if self.fp_16:
                            grad_scaler.scale(loss).backward()
                            grad_scaler.step(self.optimizer)
                            grad_scaler.update()
                            self.optimizer.zero_grad()
                        else:
                            loss.backward()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                        loss_items = torch.tensor([loss_ciou, loss_conf, loss_cls, loss])
                        mloss = (mloss * i + loss_items) / (i + 1)
                        writer.add_scalar('loss/ciou', mloss[0], epoch * len(self.train_dataloader) + i)
                        writer.add_scalar('loss/conf', mloss[1], epoch * len(self.train_dataloader) + i)
                        writer.add_scalar('loss/cls', mloss[2], epoch * len(self.train_dataloader) + i)
                        writer.add_scalar('loss/total', mloss[3], epoch * len(self.train_dataloader) + i)
                        pbar.set_postfix({'loss_total': '{:.3f}'.format(mloss[3])})
                    pbar.update()
            # Validating
            if epoch > self.eval_epoch and epoch % 10 == 0:
                valid_map = 0.0
                print('\n[{}/{} Epoch] Validating:'.format(epoch + 1, self.epochs))
                with torch.no_grad():
                    aps, inference_time = Evaluator(self.yolov4).APs_voc()
                    for i in aps:
                        print('[{}/{} Epoch] Validating: {} AP: {:.4f}'.format(epoch + 1, self.epochs, i, aps[i]))
                        valid_map += aps[i]
                    valid_map = valid_map / self.classes_num
                    print(
                        '[{}/{} Epoch] Validating: mAP: {:.4f}\n[{}/{} Epoch] Validating: Inference Time: {:.2f} ms'.format(
                            epoch + 1, self.epochs, valid_map, epoch + 1, self.epochs, inference_time))
                    writer.add_scalar('mAP/total', valid_map, epoch)
                    self.__save_model_weights(epoch, valid_map)
                writer.close()
        print('[Training Finished] Best mAP: {}'.format(self.best_mAP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='whither use GPU(0) or CPU(-1)')
    parser.add_argument('-m', '--fp_16', type=bool, default=False, help='whither to use fp16 precision')
    opt = parser.parse_args()
    Trainer(gpu_id=opt.gpu_id, fp_16=opt.fp_16).train()
