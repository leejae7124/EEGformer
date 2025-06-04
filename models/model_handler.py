import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from models.EEGDeformer2 import Deformer
from models.EEGNet import eegNet
from models.TSception import TSception
from models.EEGViT import EEGViT
from models.LGGNet import LGGNet
from models.conformer import Conformer
import os.path as osp
from utils import get_channel_info
import torchmetrics


def init_model(args): #모델 생성 함수
    model = None
    #인자로 받은 args.model 값에 따라, 다양한 모델 중 하나를 초기화하여 반환한다.
    #포함된 모델 종류: Deformer, TSception, LGGNet, EEGNet, Conformer, EEGViT
    if args.model == 'Deformer':
        model = Deformer(
            num_chan=args.num_chan, num_time=args.num_time, temporal_kernel=args.kernel_length, num_kernel=args.T,
            num_classes=args.num_class, depth=int(args.num_layers-2), heads=args.AT,
            mlp_dim=args.AT, dim_head=args.AT, dropout=args.dropout)

    #### baselines ####
    if args.model == 'TSception':
        model = TSception(
            num_classes=args.num_class, input_size=(1, args.num_chan, args.num_time),
            sampling_rate=args.sampling_rate, num_T=15, num_S=15,
            hidden=32, dropout_rate=args.dropout)
    if args.model == 'LGGNet':
        load_path = osp.join(args.load_path, 'data_{}_{}_{}'.format(args.data_format, args.dataset, args.label_type))
        _, idx_graph = get_channel_info(load_path=load_path, graph_type=args.graph_type)
        model = LGGNet(
            num_classes=args.num_class, input_size=(1, args.num_chan, args.num_time),
            sampling_rate=args.sampling_rate, num_T=args.T,
            out_graph=64, dropout_rate=args.dropout,
            pool=16, pool_step_rate=0.25, idx_graph=idx_graph
        ) #그래프 기반 모델인 LGGNet은 그래프 인덱스 정보를 추가로 필요로 함?

    if args.model == 'EEGNet':
        model = eegNet(
            nChan=args.num_chan, nTime=args.num_time, nClass=args.num_class,
            dropoutP=args.dropout, F1=8, D=2,
            C1=64)

    if args.model == 'Conformer':
        if args.num_time == 384:
            n_hidden = 800
        if args.num_time == 800:
            n_hidden = 1880
        if args.num_time == 2000:
            n_hidden = 5080
        model = Conformer(n_classes=args.num_class, n_chan=args.num_chan, n_hidden=n_hidden)

    if args.model == 'EEGViT':
        if args.num_time == 384:
            n_patch = 24
        if args.num_time == 800:
            n_patch = 40
        if args.num_time == 2000:
            n_patch = 40
        model = EEGViT(
            num_chan=args.num_chan, num_time=args.num_time, num_patches=n_patch,
            num_classes=args.num_class)

    return model


class DLModel(pl.LightningModule):
    def __init__(self, config):
        super(DLModel, self).__init__()
        self.save_hyperparameters()
        self.net = init_model(config) #실제 모델 불러옴
        self.test_step_pred = []
        self.test_step_ground_truth = []
        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=config.num_class) #평가지표(metric) 객체 생성
        self.F1 = torchmetrics.F1Score(task="multiclass", num_classes=config.num_class, average='macro') #평가지표(metric) 객체 생성
        self.config = config

    def forward(self, x): #입력 x를 받아 실제 모델에 넣어 예측 결과 반환
        return self.net(x)

    def get_metrics(self, pred, y):
        acc = self.acc(pred, y)
        f1 = self.F1(pred, y) #예측 값과 실제 라벨을 비교해 정확도와 F1 score 계산
        return acc, f1

    def training_step(self, batch, batch_nb): #학습 데이터를 모델에 통과시켜 예측 값 생성
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y) #손실 함수: Cross Entropy Loss
        acc, f1 = self.get_metrics(y_hat, y)
        self.log_dict(
            {"train_loss": loss, "train_acc": acc, "train_f1": f1},
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc, f1 = self.get_metrics(y_hat, y)
        self.log_dict(
            {"val_loss": loss, "val_acc": acc, "val_f1": f1},
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.test_step_pred.append(y_hat)
        self.test_step_ground_truth.append(y)
        acc, f1 = self.get_metrics(y_hat, y)
        self.log_dict(
            {"test_loss": loss, "test_acc": acc, "test_f1": f1},
            on_epoch=True, prog_bar=True, logger=True
        )
        return {"test_loss": loss, "test_acc": acc, "test_f1": f1}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=1e-5) #optimizer: Adam
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs, 0
        ) #learning rate scheduler: Cosine Annealing (점점 학습률을 줄이는 방식)
        return [optimizer], [scheduler]

