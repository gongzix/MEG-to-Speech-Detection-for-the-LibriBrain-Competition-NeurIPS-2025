'''
@File    :   submission.py
@Time    :   2025/06/09 10:51:47
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
from config import BASE_PATH, SENSORS_SPEECH_MASK, BATCH_SIZE, TMIN, TMAX, LABEL_WINDOW, STRIDE
from dataset import FilteredDataset
import numpy as np
from pnpl.datasets.libribrain2025.speech_dataset_holdout import LibriBrainSpeechHoldout

total_len = 560638

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
    plt.savefig(os.path.join(RESULTS_DIR, "auc_roc.png"), dpi=300)

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
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)


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

def custom_collate(batch):
    # 过滤掉None值（如果有）
    batch = [b for b in batch if b is not None]
    
    if len(batch) == 0:
        return None
    
    # 处理单个张量的情况
    if isinstance(batch[0], torch.Tensor):
        try:
            return torch.stack(batch)
        except RuntimeError:
            # 如果stack失败，尝试使用默认collate
            return torch.utils.data.default_collate(batch)
    
    # 处理元组或列表的情况
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]
    
    # 其他情况使用默认collate
    else:
        return torch.utils.data.default_collate(batch)
    

class Custom_HoldoutDataset(LibriBrainCompetitionHoldout):
    def __init__(self, data_path,
                 tmin: float = 0.0,
                 tmax: float = 0.8,
                 standardize=True,
                 channel_means=None,
                 channel_stds=None,
                 clipping_boundary=10,
                 stride=1,
                 task: str = "speech",
                 download: bool = True):
        
        self.data_path = data_path
        self.task = task
        self.dataset = None
        if task == "speech":
            try:
                self.dataset = LibriBrainSpeechHoldout(
                    data_path=self.data_path,
                    tmin = tmin,
                    tmax = tmax,
                    include_run_keys=[("0", "2025", "COMPETITION_HOLDOUT", "1")],
                    standardize=standardize,
                    clipping_boundary=clipping_boundary,
                    preprocessing_str="bads+headpos+sss+notch+bp+ds",
                    channel_means =channel_means,
                    channel_stds = channel_stds,
                    preload_files=False,
                    include_info=True,
                    stride=stride,
                    download=download
                )
                self.samples = self.dataset.samples
            except Exception as e:
                warnings.warn(f"Failed to load speech dataset: {e}")
                raise RuntimeError("Failed to load speech dataset. Check the data path and parameters.")
        if task == "phoneme":
            raise NotImplementedError(f"Task '{task}' is not supported yet.")
        self.tmin = tmin
        self.tmax = tmax
        
    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        fixed_length = int(250*(self.tmax-self.tmin))
        if data.shape[-1] < fixed_length:
            data = torch.nn.functional.pad(data, (0, fixed_length - data.shape[-1]))
        # elif data.shape[-1] > fixed_length:
        #     data = data[..., :fixed_length]

        return data.clone()# 避免共享 storage    

def remove_short_segments(pred, min_len=15, target_value=1):
    """
    删除持续时间小于 min_len 的 target_value 区段
    pred: np.array of shape [T], binary (0 or 1)
    min_len: 最小保留长度（单位：帧）
    target_value: 要处理的值，1表示清理短语音段，0表示清理短静默段
    """
    pred = pred.copy()
    T = len(pred)
    i = 0
    while i < T:
        if pred[i] == target_value:
            start = i
            while i < T and pred[i] == target_value:
                i += 1
            length = i - start
            if length < min_len:
                pred[start:i] = 1 - target_value  # 抹掉短段
        else:
            i += 1
    return pred       


    
if __name__ == "__main__":
    L.seed_everything(42)
    num_workers = 8
    task="speech"
    channel_means = np.load(f"{BASE_PATH}/train_data_channel_means.npy")
    channel_stds = np.load(f"{BASE_PATH}/train_data_channel_stds.npy")       
    # ========== 新增部分：生成 submission CSV 文件 ==========
    speech_holdout_dataset = Custom_HoldoutDataset(
        data_path=f"{BASE_PATH}",
        tmin=TMIN,  # Same as in the other LibriBrain dataset - this is where we'll store the data
        tmax=TMAX,  # Also identical to the other datasets - how many samples to return group together (e.g., if combining multiple samples).
        task="speech",  # "speech" or "phoneme" ("phoneme" is not supported until the launch of the Phoneme Classification track!)    
        download=False,
        standardize=True,        
        channel_means=channel_means,
        channel_stds=channel_stds,                
    )

    # Next, create a DataLoader for the dataset
    # on_hold_loader = DataLoader(speech_holdout_dataset, batch_size=4096, shuffle=False, num_workers=16, drop_last=False, persistent_workers=True, collate_fn=custom_collate)
    on_hold_loader = DataLoader(speech_holdout_dataset, batch_size=256, shuffle=False, num_workers=16, drop_last=False, persistent_workers=True)
    # The final array of predictions must contain len(speech_holdout_dataset) values between 0..1
    segments_to_predict = len(speech_holdout_dataset)
    print("segments_to_predict", segments_to_predict)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = SpeechClassifier.load_from_checkpoint(
        checkpoint_path='/cpfs04/user/gongzixuan/PNPL/codev3/scripts/L_checkpoints/bitcn-20--epoch=4-val_loss=0.31.ckpt',
        input_dim=INPUT_DIM,
        model_dim=MODEL_DIM,
        learning_rate=LEARNING_RATE,
        dropout_rate=DROPOUT_RATE,
        lstm_layers=LSTM_LAYERS,
        weight_decay=WEIGHT_DECAY,
        batch_norm=BATCH_NORM,
        bi_directional=BI_DIRECTIONAL
    )

    model.eval()
    all_y_probs = []

    # with torch.no_grad():
    #     for batch in tqdm(on_hold_loader):
    #         # print("batch", batch.shape)
    #         batch = batch.to(model.device)  # 确保数据在相同设备
    #         logits = model(batch)
    #         probs = torch.sigmoid(logits)            
    #         all_y_probs.append(probs)

    # from scipy.ndimage import median_filter

             
    # y_probs = torch.cat(all_y_probs).cpu().numpy()
    # # y_pred = (y_probs >= 0.5).astype(int)
    # # y_pred = remove_short_segments(y_pred, min_len=25, target_value=1)
    # #y_pred = remove_short_segments(y_pred, min_len=10, target_value=0)

    # # 中值滤波 (窗口大小设为5)
    # smoothed_probs = median_filter(y_probs, size=40, mode='nearest')
    # y_pred = (smoothed_probs >= 0.5).astype(int)   

    # print("y_pred", y_pred)
    center_N = LABEL_WINDOW
    T_out = 2 * center_N  # 输出帧数 = 11
    # 设置标准差 σ
    sigma = center_N / 2
    # 高斯权重分布
    positions = np.arange(T_out)  # [0, 1, ..., 10]
    weights = np.exp(-((positions - center_N) ** 2) / (2 * sigma ** 2))
    weights = weights / weights.sum()  # 归一化为总和为 1 

    frame_probs = np.zeros(total_len)
    frame_counts = np.zeros(total_len)
    start_indices_array = np.arange(0, total_len - 250*(TMAX-TMIN) + 1, 1)
    num = 0
    
    # on_hold_loader 滑动投票
    with torch.no_grad():
        for batch in tqdm(on_hold_loader):
            batch = batch.to(model.device)
            logits = model(batch)                      # [B，t]
            probs = torch.sigmoid(logits).cpu().numpy()  # [B, t]
            start_indices = start_indices_array[num*batch.shape[0]:(num+1)*batch.shape[0]]

            for i, start in enumerate(start_indices):
                center = int(start + 250*(TMAX-TMIN) // 2)
                for offset in range(-center_N, center_N):
                    frame = center + offset
                    if 0 <= frame < total_len:
                        frame_probs[frame] += (probs[i][offset + center_N].item())
                        frame_counts[frame] += 1
            num = num + 1
    y_probs = frame_probs / (frame_counts + 1e-8)
    #y_pred = (y_probs >= 0.5).astype(int)

    # 中值滤波 (窗口大小设为5)
    from scipy.ndimage import median_filter
    smoothed_probs = median_filter(y_probs, size=40, mode='nearest')
    y_pred = (smoothed_probs >= 0.5).astype(int) 

    print("y_pred", y_pred)

    #使用 dataset 自带方法生成 csv
    model_name = "bitcn-L-20"
    csv_name = f"{model_name}_s{STRIDE}_{LABEL_WINDOW}.csv"
    prob_name = f"{model_name}_s{STRIDE}_{LABEL_WINDOW}_prob.csv"

    generate_submission_in_csv(
        task,
        y_pred,
        os.path.join(RESULTS_DIR, csv_name)
    )
    print(f"Successfully Saved {csv_name}")    


    generate_submission_in_csv(
        task,
        y_probs,
        os.path.join(RESULTS_DIR + '/prob/', prob_name)
    )