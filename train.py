'''
@File    :   train.py
@Time    :   2025/06/011 21:34:19
@Author  :   Parameter Team
@Version :   1.0
'''



import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint 
from torch.utils.data import DataLoader
from model import SpeechModel, BCEWithLogitsLossWithSmoothing
from dataset import get_data_loaders
from config import *
import torch
from model import SpeechClassifier
from lightning.pytorch.loggers import WandbLogger  # 替换掉原来的 CSVLogger 即可
from torch.optim.lr_scheduler import OneCycleLR
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    args = parser.parse_args()
 
    L.seed_everything(42)

    train_loader, val_loader, test_loader = get_data_loaders(globals())

    model = SpeechClassifier(
        input_dim=INPUT_DIM,
        model_dim=MODEL_DIM,
        learning_rate=LEARNING_RATE,
        dropout_rate=DROPOUT_RATE,
        lstm_layers=LSTM_LAYERS,
        weight_decay=WEIGHT_DECAY,
        batch_norm=BATCH_NORM,
        bi_directional=BI_DIRECTIONAL,
        depth=2
    )
    #model = SpeechClassifier.load_from_checkpoint('./checkpoints/tcn-s1--epoch=05-val_loss=0.36.ckpt')
    
    # 使用 WandB Logger
    wandb_logger = WandbLogger(project="VAD-final check", log_model=False)

    # 设置模型检查点回调，监控验证集损失
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,                # 只保存最好的一个模型
        dirpath="L_checkpoints/",      # 保存路径
        filename="tcn-20--{epoch}-{val_loss:.2f}",  # 文件名格式
        #every_n_train_steps=5000,
        every_n_epochs=1,            # 最大规模训练时，每个epoch都保存
        save_weights_only=False,      # 保存整个模型而不仅仅是权重
        #save_on_train_epoch_end=False,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=100, mode="min")

    trainer = L.Trainer(
        devices="auto",
        max_epochs=200,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],  # 添加检查点回调
        enable_checkpointing=True,
        strategy="ddp",
        #log_every_n_steps=1000,
        #val_check_interval=5000,
        log_every_n_steps=1,
        # resume_from_checkpoint=CHECKPOINT_PATH if args.resume and os.path.exists(CHECKPOINT_PATH) else None
    )

    trainer.fit(model, train_loader, val_loader)#, ckpt_path="./checkpoints/tcn-s1--epoch=05-val_loss=0.36.ckpt")
    
    # 测试阶段加载最佳模型
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        model = SpeechClassifier.load_from_checkpoint(best_model_path)
    
    trainer.test(model, test_loader)

