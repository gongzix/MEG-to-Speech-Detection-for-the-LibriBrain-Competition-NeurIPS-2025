'''
@File    :   dataset.py
@Time    :   2025/06/09 10:52:14
@Author  :   Parameter Team
@Version :   1.0
'''



import random
from torch.utils.data import DataLoader, Dataset
from pnpl.datasets import LibriBrainSpeech
from config import BASE_PATH, SENSORS_SPEECH_MASK, BATCH_SIZE, NUM_WORKERS, TMIN, TMAX, LABEL_WINDOW, STRIDE
from pnpl.datasets.libribrain2025.constants import RUN_KEYS, VALIDATION_RUN_KEYS, TEST_RUN_KEYS
from braindecode.augmentation import AugmentedDataLoader, SignFlip, FrequencyShift
from braindecode.augmentation import Compose
import numpy as np

# class FilteredDataset(Dataset):
#     def __init__(self, dataset, limit_samples=None, apply_sensors_speech_mask=True):
#         self.dataset = dataset
#         self.limit_samples = limit_samples
#         self.apply_sensors_speech_mask = apply_sensors_speech_mask
#         self.sensors_speech_mask = SENSORS_SPEECH_MASK  # From config.py
#         self.balanced_indices = list(range(len(dataset.samples)))
#         self.balanced_indices = random.sample(self.balanced_indices, len(self.balanced_indices))

#     def __len__(self):
#         return self.limit_samples or len(self.balanced_indices)

#     def __getitem__(self, index):
#         try:
#             original_idx = self.balanced_indices[index]
#             sensors = self.dataset[original_idx][0][self.sensors_speech_mask] \
#                 if self.apply_sensors_speech_mask else self.dataset[original_idx][0][:]
#             label_from_the_middle_idx = self.dataset[original_idx][1].shape[0] // 2
#         except Exception as e:
#             print(f"Error at index {index}: {e}")
#             raise        
#         return sensors, self.dataset[original_idx][1][label_from_the_middle_idx]


class FilteredDataset(Dataset):
    def __init__(self, dataset, limit_samples=None, apply_sensors_speech_mask=True):
        self.dataset = dataset
        self.limit_samples = limit_samples
        self.apply_sensors_speech_mask = apply_sensors_speech_mask
        self.sensors_speech_mask = SENSORS_SPEECH_MASK  # From config.py
        self.balanced_indices = list(range(len(dataset.samples)))
        self.balanced_indices = random.sample(self.balanced_indices, len(self.balanced_indices))
        
        # 添加 transform 属性
        self.transform = None  # 允许外部赋值

    def __len__(self):
        return self.limit_samples or len(self.balanced_indices)

    # def __getitem__(self, index):
    #     try:
    #         original_idx = self.balanced_indices[index]
    #         sensors = self.dataset[original_idx][0][self.sensors_speech_mask] \
    #             if self.apply_sensors_speech_mask else self.dataset[original_idx][0][:]
    #         label_from_the_middle_idx = self.dataset[original_idx][1].shape[0] // 2
    #     except Exception as e:
    #         print(f"Error at index {index}: {e}")
    #         raise
        
    #     # 在这里应用 transform
    #     if self.transform is not None:
    #         sensors = self.transform(sensors)
        
    #     return sensors, self.dataset[original_idx][1][label_from_the_middle_idx]
    
    def __getitem__(self, index):
        try:
            original_idx = self.balanced_indices[index]
            sensors = self.dataset[original_idx][0][self.sensors_speech_mask] \
                if self.apply_sensors_speech_mask else self.dataset[original_idx][0][:]
            label = self.dataset[original_idx][1]
            label_from_the_middle_idx = label.shape[0] // 2
            label_window = LABEL_WINDOW
            label_single = label#[label_from_the_middle_idx-label_window:label_from_the_middle_idx+label_window]#.float().mean()
            #label_single = label[label_from_the_middle_idx]
            #label_single = label

        except Exception as e:
            print(f"Error at index {index}: {e}")
            raise
        # print("sensors", sensors.shape)
        # print("label", label.shape)
        # 在这里应用 transform，让它处理 input 和 label
        if self.transform is not None:
            sensors_aug, label_single_aug = self.transform(sensors, label_single)
            return sensors_aug, label_single_aug
        else:
            return sensors, label_single
        # return sensors, label_single, sensors_aug, label_single_aug
        
class SignalOnlyTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal, label):
        for t in self.transforms:
            signal = t(signal)
        return signal, label

def get_data_loaders(config):
    print(RUN_KEYS)  # Print all available keys
    print(VALIDATION_RUN_KEYS) # ('0', '11', 'Sherlock1', '2') is the validation split
    print(TEST_RUN_KEYS)  # [('0', '12', 'Sherlock1', '2')] is the test split
    #print([k for k in RUN_KEYS if k not in {*VALIDATION_RUN_KEYS, *TEST_RUN_KEYS}])  # All other keys are the train split    

    

    # Train data
    train_run_keys = [("0", str(i), "Sherlock1", "1") for i in range(1, 11)] + \
                    [("0", str(i), "Sherlock2", "1") for i in range(1, 13) if i != 2] + \
                    [("0", str(i), "Sherlock3", "1") for i in range(1, 13)] + \
                    [("0", str(i), "Sherlock4", "1") for i in range(1, 13)]+ \
                    [("0", str(i), "Sherlock5", "1") for i in range(1, 16)] + \
                    [("0", str(i), "Sherlock6", "1") for i in range(1, 15)] + \
                    [("0", str(i), "Sherlock7", "1") for i in range(1, 15)]
                    #[("0", "11", "Sherlock1", "2")] + \
                    #[("0", "12", "Sherlock1", "2")]
                    
                     
    # train_data = LibriBrainSpeech(data_path=f"{BASE_PATH}", include_run_keys=train_run_keys, tmin=0.0, tmax=0.8, preload_files=True)
        
    # train_data = LibriBrainSpeech(data_path=f"{BASE_PATH}", tmin=0.0, tmax=0.8, preload_files=True)
    # train_data_filtered = FilteredDataset(train_data)
    # train_loader = DataLoader(train_data_filtered, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, persistent_workers=True)
    # print(f"Train data contains {len(train_data_filtered)} samples")

    # 定义你的数据增强方法
    freq_shift = FrequencyShift(
        probability=.1,
        sfreq=250,  # 确保这里你已经定义了 sfreq 变量
        max_delta_freq=2.  # 频率偏移范围在 -2 到 2 Hz 之间
    )

    sign_flip = SignFlip(probability=.1)
    transforms = [
        freq_shift,
        sign_flip
    ]

    # 加载数据
    train_data = LibriBrainSpeech(data_path=f"{BASE_PATH}", standardize=True, include_run_keys=train_run_keys, tmin=TMIN, tmax=TMAX, preload_files=True,stride=STRIDE)
    train_data_filtered = FilteredDataset(train_data)    
    #composed_transforms = Compose(transforms=transforms)  
    composed_transforms = SignalOnlyTransform(transforms=transforms)  
    train_data_filtered.transform = composed_transforms      
    train_loader = DataLoader(train_data_filtered, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=4, pin_memory=True, persistent_workers=True)
    print(f"Train data contains {len(train_data_filtered)} samples")
    #print(train_data.channel_means)
    #print(train_data.channel_stds)
    # Save the channel means and standard deviations
    
    np.save(f"{BASE_PATH}/train_data_channel_means.npy", train_data.channel_means)
    np.save(f"{BASE_PATH}/train_data_channel_stds.npy", train_data.channel_stds)
        
    # x, y, x_aug, y_aug = train_data_filtered[0]
    # print("Sample shape before transform:", x.shape)
    # if train_data_filtered.transform:        
    #     print("Sample shape after transform:", x_aug.shape)
    #     print("Difference:", (x != x_aug).any())


    # Val data
    val_data = LibriBrainSpeech(data_path=f"{BASE_PATH}", include_run_keys=[("0", "11", "Sherlock1", "2")], standardize=True, 
                                    channel_means=train_data.channel_means,
                                    channel_stds=train_data.channel_stds,
                                    tmin=TMIN, tmax=TMAX, preload_files=True, stride=STRIDE)
    val_data_filtered = FilteredDataset(val_data)
    val_loader = DataLoader(val_data_filtered, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=4, pin_memory=True, persistent_workers=True)
    print(f"Validation data contains {len(val_data_filtered)} samples")
    
    # Test data
    test_data = LibriBrainSpeech(data_path=f"{BASE_PATH}", include_run_keys=[("0", "12", "Sherlock1", "2")], standardize=True, 
                                    channel_means=train_data.channel_means,
                                    channel_stds=train_data.channel_stds,                                 
                                    tmin=TMIN, tmax=TMAX, preload_files=True, stride=STRIDE)
    test_data_filtered = FilteredDataset(test_data)
    test_loader = DataLoader(test_data_filtered, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=4, pin_memory=True, persistent_workers=True)
    print(f"Test data contains {len(test_data_filtered)} samples\n")
    
    # Let's look at the first batch:
    first_batch = next(iter(train_loader))
    inputs, labels = first_batch
    print("Batch input shape:", inputs.shape)
    print("Batch label shape:", labels.shape)    


    
    return train_loader, val_loader, test_loader
