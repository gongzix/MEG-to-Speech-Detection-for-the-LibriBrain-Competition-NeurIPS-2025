'''
@File    :   evaluate.py
@Time    :   2025/06/09 10:52:04
@Author  :   Parameter Team
@Version :   1.0
'''


import lightning as L
from sklearn.metrics import roc_curve, auc, jaccard_score
import matplotlib.pyplot as plt
import torch
from model import SpeechClassifier
from dataset import get_data_loaders
from config import *
import os
from torch.utils.data import DataLoader
from pnpl.datasets import LibriBrainCompetitionHoldout
from tqdm import tqdm
import os
from torch.utils.data import Dataset
import torch
from pnpl.datasets.libribrain2025.constants import PHONEME_CLASSES, PHONEME_HOLDOUT_PREDICTIONS
import csv
import torch
import warnings
import random
from torch.utils.data import DataLoader, Dataset
from pnpl.datasets import LibriBrainSpeech
from config import BASE_PATH, SENSORS_SPEECH_MASK, BATCH_SIZE, TMIN, TMAX, STRIDE
from dataset import FilteredDataset
import numpy as np

def plot_auc_roc(labels, probs):
    labels_flat = labels.flatten()
    probs_flat = probs.flatten()
    fpr, tpr, _ = roc_curve(labels_flat, probs_flat)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    #plt.savefig(os.path.join(RESULTS_DIR, "auc_roc.png"), dpi=300)

    print(f"\nAUC is {roc_auc}")
    return roc_auc


def plot_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white')
    plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()
    #plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)


    import numpy as np
    print("Label 0 count:", np.sum(y_true == 0))
    print("Label 1 count:", np.sum(y_true == 1))

    # 2. 打印分类报告
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # 3. 显式计算 Macro F1-score 和 Micro F1-score
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")

    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Micro F1-score: {micro_f1:.4f}")

    # 4. 如果需要，可以手动计算 Precision 和 Recall
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = (2 * precision * recall) / (precision + recall)
    print(f"F1-score (Manual Calculation): {f1:.4f}")

def generate_submission_in_csv(task, predictions, output_path: str):
    """
    Generates a submission file in CSV format for the LibriBrain competition.
    The file contains the run keys and the corresponding labels.
    Args:
        predictions (List[Tensor]): List of scalar tensors, each representing a speech probability.
        output_path (str): Path to save the CSV file.
    """
    if task == "speech":      
        with open(output_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["segment_idx", "speech_prob"])

            for idx, tensor in enumerate(predictions):
                # Ensure we extract the scalar float from tensor
                speech_prob = tensor.item() if isinstance(
                    tensor, torch.Tensor) else float(tensor)
                writer.writerow([idx, speech_prob])
    elif task == "phoneme":
        if len(predictions) != PHONEME_HOLDOUT_PREDICTIONS:
            raise (ValueError(
                "Length of Phonemes predictions does not match number of segments."))
        if predictions[0].shape[0] != PHONEME_CLASSES:
            raise (ValueError(
                f"Phonemes classes does not match expect size ({PHONEME_CLASSES})."))
        with open(output_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Create header: segment_idx, phoneme_1, ..., phoneme_39
            header = ["segment_idx"] + \
                [f"phoneme_{i + 1}" for i in range(39)]
            writer.writerow(header)

            for idx, tensor in enumerate(predictions):
                # Ensure tensor is a flat list of floats
                if isinstance(tensor, torch.Tensor):
                    probs = tensor.squeeze().tolist()  # shape: (39,)
                else:
                    # if tensor is already a list-like
                    probs = list(tensor)

                writer.writerow([idx] + probs)
    
    
if __name__ == "__main__":
    L.seed_everything(42)
    num_workers = 4
    task="speech"
    # _, val_loader, test_loader = get_data_loaders(globals())
    channel_means = np.load(f"{BASE_PATH}/train_data_channel_means.npy")
    channel_stds = np.load(f"{BASE_PATH}/train_data_channel_stds.npy")    
    # Test data
    test_data = LibriBrainSpeech(data_path=f"{BASE_PATH}", include_run_keys=[("0", "12", "Sherlock1", "2")], standardize=True, 
                                channel_means=channel_means,
                                channel_stds=channel_stds,                                 
                                 tmin=TMIN, tmax=TMAX, preload_files=True)
    test_data_filtered = FilteredDataset(test_data)
    test_loader = DataLoader(test_data_filtered, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, persistent_workers=True)
    print(f"Test data contains {len(test_data_filtered)} samples\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = SpeechClassifier.load_from_checkpoint(
        checkpoint_path="/cpfs04/user/gongzixuan/PNPL/codev3/scripts/L_checkpoints/bitcn-20--epoch=4-val_loss=0.31.ckpt",
        input_dim=INPUT_DIM,
        model_dim=MODEL_DIM,
        learning_rate=LEARNING_RATE,
        dropout_rate=DROPOUT_RATE,
        lstm_layers=LSTM_LAYERS,
        weight_decay=WEIGHT_DECAY,
        batch_norm=BATCH_NORM,
        bi_directional=BI_DIRECTIONAL
    )

    trainer = L.Trainer(devices="auto")
    trainer.test(model, dataloaders=test_loader)

    model.eval()
    all_y_true = []
    all_y_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            x, y = batch
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_y_true.append(y.cpu())
            all_y_probs.append(probs.cpu())

    y_true = torch.cat(all_y_true, dim=0)
    y_probs = torch.cat(all_y_probs, dim=0)

    y_true = y_true.view(-1).numpy().astype(int)       # shape [N*12]
    y_probs = y_probs.view(-1).numpy()                 # shape [N*12]

    threshold = [0.5]
    for t in threshold:
        y_pred = (y_probs >= t).astype(int)
        y_true = (y_true >= t).astype(int)
        plot_confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(y_true, y_pred)

        iou = jaccard_score(y_true, y_pred, average="binary")
        print(f"threshold:{t}\nIoU (Jaccard Index): {iou}")
    
    # ========== 新增部分：生成 submission CSV 文件 ==========
    #print("y_pred", y_pred)
    # 使用 dataset 自带方法生成 csv
    
    generate_submission_in_csv(
        task,
        y_pred,
        os.path.join(RESULTS_DIR, "test_speech_predictions.csv")
    )
     