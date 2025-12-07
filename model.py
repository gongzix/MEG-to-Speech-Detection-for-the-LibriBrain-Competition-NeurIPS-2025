'''
@File    :   model.py
@Time    :   2025/06/09 10:53:12
@Author  :   Parameter Team
@Version :   1.0
'''


import torch
import torch.nn as nn
import lightning as L
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('/cpfs04/user/gongzixuan/PNPL/codev3')
import numpy as np
from config import SAMPLES, TMIN, TMAX, LABEL_WINDOW, WARM_UP
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import math
from transformers import get_cosine_schedule_with_warmup


class MultiScaleTCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sizes=[3, 5, 7], dilations=[1, 2, 4], dropout=0.2):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), "kernel_sizes and dilations must be same length"

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=(k - 1) * d // 2, dilation=d),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for k, d in zip(kernel_sizes, dilations)
        ])
        self.fuse = nn.Conv1d(out_ch * len(kernel_sizes), out_ch, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x):  # x: [B, C, T]
        outs = [branch(x) for branch in self.branches]  # List of [B, out_ch, T]
        x = torch.cat(outs, dim=1)                      # [B, out_ch * num_branches, T]
        x = self.fuse(x)                                # [B, out_ch, T]
        return self.activation(x)


class MultiScaleTCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_blocks=3, 
                 kernel_sizes=[3, 5, 7], dilations=[64, 128, 256], dropout_rate=0.2, batch_norm = False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, padding=1)
        self.batch_norm = nn.BatchNorm1d(num_features=hidden_channels) if batch_norm else nn.Identity()
        self.conv_dropout = nn.Dropout(p=dropout_rate)

        self.blocks = nn.Sequential(*[
            MultiScaleTCNBlock(hidden_channels, hidden_channels, kernel_sizes, dilations, dropout_rate)
            for _ in range(num_blocks)
        ])

        # 输出用于分类或后续接 transformer/vad 等结构
        self.output_proj = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x):  # x: [B, C, T]
        x = self.conv(x)        
        x = self.batch_norm(x)
        x = self.blocks(x)       # [B, hidden_channels, T]
        x = self.output_proj(x)  # [B, hidden_channels, T]
        x = self.pool(x).squeeze(-1)  # [B, C]
        return self.fc(x)  # [B, 1]

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(p=dropout)

        self.downsample = nn.Sequential()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        return self.relu(out + identity)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(p=dropout)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(p=dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        return self.relu(out + residual)

class CNN_TCN(nn.Module):
    def __init__(self, input_dim=64, res_channels=128, tcn_channels=128, tcn_layers=2, num_classes=1, dropout=0.1, batch_norm=False):
        super().__init__()

        #self.input_dim = input_dim
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=res_channels, kernel_size=1, padding=0)
        self.batch_norm = nn.BatchNorm1d(num_features=res_channels) if batch_norm else nn.Identity()
        self.conv_dropout = nn.Dropout(p=dropout)

        tcn_blocks = []
        for i in range(tcn_layers):
            dilation = 2 ** i
            #dilation = 2 + i #linear dilation
            #dilation = 2 + 2*i
            tcn_blocks.append(
                TCNBlock(res_channels, tcn_channels, dilation=dilation, dropout=dropout)
            )
        self.tcn = nn.Sequential(*tcn_blocks)
        self.fc = nn.Linear(tcn_channels, 1)

    def forward(self, x):  # x: [B, C, T]

        x = self.conv(x)        
        x = self.batch_norm(x)
        x = self.conv_dropout(x)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x).squeeze(-1)
        return x

        #return output



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # [D/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, T, D]

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :].to(x.device)


class TransformerVAD(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=4,dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)  # C -> d_model
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True,norm_first=True, dropout=dropout,dim_feedforward=d_model*4,)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 1),
        )

    def forward(self, x):  # x: [B, C, T]
        x = x.permute(0, 2, 1)  # -> [B, T, C]
        x = self.input_proj(x)  # -> [B, T, d_model]
        x = self.pos_enc(x)     # 加入位置编码
        x = self.transformer(x) # -> [B, T, d_model]
        #x = x.mean(dim=1)       # 平均池化：获得序列级特征 [B, d_model]
        out = self.classifier(x).squeeze(-1)  # -> [B, 1]
        return out

class SpeechModel(nn.Module):
    def __init__(self, input_dim, model_dim, dropout_rate=0.3, lstm_layers=1, bi_directional=False, batch_norm=False, contrastive = True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=model_dim, kernel_size=1, padding=0)
        self.batch_norm = nn.BatchNorm1d(num_features=model_dim) if batch_norm else nn.Identity()
        self.conv_dropout = nn.Dropout(p=dropout_rate)

        
        self.lstm = nn.LSTM(
            input_size=model_dim,
            hidden_size=model_dim,
            num_layers=lstm_layers,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=bi_directional
        )
        '''
        self.lstm = ResidualLSTM(
            input_size=model_dim, 
            hidden_size=model_dim, 
            num_layers=lstm_layers,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=bi_directional
        )
        '''
        
        self.lstm_dropout = nn.Dropout(p=dropout_rate)
        self.speech_classifier = nn.Linear(model_dim, LABEL_WINDOW*2)

        #self.positions = nn.Parameter(torch.zeros(input_dim)) #Pos_emb
        self.input_dim = input_dim
        self.contrastive = contrastive
        '''
        self.blocks = nn.ModuleList([
            FilterBlock(
                model_dim = model_dim, drop=dropout_rate, time_length = 200)
            for i in range(1)])
        '''

    def forward(self, x):
        # print("x", x.shape)

        # downsampling
        # x = F.interpolate(x, size=100, mode='linear', align_corners=False)
        # x = nn.AvgPool1d(kernel_size=2, stride=2)(x)
        # pos_emb = self.positions.unsqueeze(0).unsqueeze(-1) 
        # pos_emb = pos_emb.expand(x.shape[0], self.input_dim, x.shape[2])
        # x = x + pos_emb

        x = self.conv(x)        
        x = self.batch_norm(x)
        x = self.conv_dropout(x)
        #x = self.cbam(x)

        #for blk in self.blocks:
            #x = blk(x)

        output, (h_n, _) = self.lstm(x.permute(0, 2, 1))
        h_n = self.lstm_dropout(h_n[-1] if self.lstm.num_layers > 1 else h_n)
        output = self.speech_classifier(h_n)
        #output = self.speech_classifier(output).squeeze(-1)
        if self.contrastive:
            return output, h_n
        else:
            return output


class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=1):
        super().__init__()
        self.smoothing = smoothing
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, logits, target):
        target = target.float()
        target_smoothed = target * (1 - self.smoothing) + self.smoothing * 0.5
        return self.bce_loss(logits, target_smoothed)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # 用logits更稳定

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()
def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    features: Tensor [B, D] - L2 normalized
    labels:   Tensor [B]    - binary labels: 0 or 1
    """
    device = features.device
    features = F.normalize(features, dim=1)  # cosine similarity

    similarity = torch.matmul(features, features.T)  # [B, B]
    similarity /= temperature

    labels = labels.contiguous().view(-1, 1)  # [B, 1]
    mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]，正对是1

    # 去掉对角线（自己和自己）
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
    mask = mask * logits_mask

    # log-softmax 计算
    exp_sim = torch.exp(similarity) * logits_mask
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

    loss = -mean_log_prob_pos.mean()
    return loss
    
class SpeechClassifier(L.LightningModule):
    """
    Parameters:
        input_dim (int): Number of input channels/features. This is passed to the underlying SpeechModel.
        model_dim (int): Dimensionality of the intermediate model representation.
        learning_rate (float, optional): Learning rate for the optimizer.
        weight_decay (float, optional): Weight decay for the optimizer.
        batch_size (int, optional): Batch size used during training and evaluation.
        dropout_rate (float, optional): Dropout probability applied after convolutional and LSTM layers.
        smoothing (float, optional): Label smoothing factor applied in the BCEWithLogits loss.
        pos_weight (float, optional): Weight for the positive class in the BCEWithLogits loss.
        batch_norm (bool, optional): Indicates whether to use batch normalization.
        lstm_layers (int, optional): Number of layers in the LSTM module within the SpeechModel.
        bi_directional (bool, optional): If True, uses a bidirectional LSTM in the SpeechModel; otherwise, uses a unidirectional LSTM.
    """

    def __init__(self, input_dim, model_dim, learning_rate=1e-3, weight_decay=0.01, batch_size=32, dropout_rate=0.3, smoothing=0.1, pos_weight = 1.0 , batch_norm = False, lstm_layers = 1, bi_directional = False, depth=2, contrastive = False):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.contrastive = contrastive
        #self.model = SpeechModel(input_dim, model_dim, dropout_rate=dropout_rate, lstm_layers=lstm_layers, bi_directional=bi_directional, batch_norm=batch_norm, contrastive = contrastive)
        #self.model = TransformerVAD(input_dim, model_dim, nhead=4, num_layers=4,dropout=dropout_rate)
        self.model = CNN_TCN(input_dim, res_channels=model_dim, tcn_channels=model_dim, tcn_layers=20, dropout=dropout_rate)
        self.loss_fn = BCEWithLogitsLossWithSmoothing(smoothing=smoothing, pos_weight = pos_weight)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

        self.val_step_outputs = []
        self.test_step_outputs = {}


    def forward(self, x):
            return self.model(x)

    def _shared_eval_step(self, batch, stage):
        x = batch[0]
        y = batch[1] # (batch, seq_len)

        if self.contrastive:
            logits, h = self(x)
            contrastive_loss = supervised_contrastive_loss(h, y.float().view(-1, 1))
            alpha = 0.02
            loss = alpha * contrastive_loss + self.loss_fn(logits, y.float().view(-1, 1))
        else:
            logits = self(x)
            #loss = self.focal_loss(logits, y.float().view(-1, 1))
            loss = self.loss_fn(logits, y.float().view_as(logits))#.view(-1, 1))
        # loss = self.loss_fn(logits, y.unsqueeze(1).float())
        # loss = self.loss_fn(logits, y.float())        
        probs = torch.sigmoid(logits)
        y_probs = probs.detach().cpu()

        y_true = batch[1].detach().cpu()
        meg = x.detach().cpu()

        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        return loss


    def training_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, "train")


    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, "val")


    def test_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]

        # ugly, taking care of only one label
        if len(y.shape) != 1:
            y = y.flatten(start_dim=0, end_dim=1).view(-1, 1)  # (batch, seq_len) -> (batch * seq_len, 1)
        else:
            y = y.unsqueeze(1)

        if self.contrastive:
            logits, h = self(x)
        else:
            logits = self(x)

        loss = self.loss_fn(logits, y.float().view_as(logits))
        #loss = self.focal_loss(logits, y.float())
        probs = torch.sigmoid(logits)

        # Append data to the defaultdict
        # Ensure keys exist before appending
        if "y_probs" not in self.test_step_outputs:
            self.test_step_outputs["y_probs"] = []
        if "y_true" not in self.test_step_outputs:
            self.test_step_outputs["y_true"] = []
        if "meg" not in self.test_step_outputs:
            self.test_step_outputs["meg"] = []

        # Append data
        if y.shape[-1] != 1:
            self.test_step_outputs["y_probs"].extend(
                probs.detach().view(x.shape[0], x.shape[-1]).cpu())  # (batch, seq_len)
        else:
            self.test_step_outputs["y_probs"].extend(
                probs.detach().view(x.shape[0], probs.shape[-1]).cpu())  # (batch, seq_len)

        self.test_step_outputs["y_true"].extend(batch[1].detach().cpu())  # (batch, seq_len)
        self.test_step_outputs["meg"].extend(x.detach().cpu())  # MEG data (batch, channels, seq_len)

        return self._shared_eval_step(batch, "test")

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if WARM_UP:
            scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=294600
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # 每一步更新（不是每个epoch）
                    "frequency": 1
                }
            }
        else:
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5),
                'monitor': 'val_loss',  # 告诉 Lightning 要监控哪个指标
                'interval': 'epoch',    # 每个 epoch 检查一次
                'frequency': 1,
                "name": "learning_rate",
            }
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
        }
        
    
    
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        scheduler = self.lr_schedulers()
        current_lr = scheduler.get_last_lr()[0]
        self.log("lr", current_lr, prog_bar=True, on_step=True, on_epoch=False, logger=True)

    '''
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100000,   # 每 10000 step 降一次学习率
            gamma=0.5          # 乘以 0.5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',   # <<< 按 step 调整学习率！
                'frequency': 1,
                'name': "learning_rate"
            }
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': OneCycleLR(
                optimizer, 
                max_lr = self.learning_rate,
                steps_per_epoch = SAMPLES // self.batch_size,
                epochs = self.epochs
            ),
            'interval': 'step'
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    '''
