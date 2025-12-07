This is the code implementation by **Parameter Team** for the Standard Track of the MEG Speech Detection in the first phase of [LibriBrain Competition](https://neural-processing-lab.github.io/2025-libribrain-competition/) (**NeurIPS 2025 Competition Track**).

**2025-12-7**：🎉 Congratulations! Our model achieved **1st** place in the Speech Detection Standard Track! 🥇🏆

![track1](assets/sherlock3.gif)

## Note:

In practice, we found that the choice of model architecture was not the dominant factor. Among the models we tested, the classical CNN+TCN (Temporal Convolutional Network) achieved the best single-model performance, with deeper architectures further improving results on the holdout dataset. In addition, CLDNN, Transformer, and Transformer-inspired EEG models yielded suboptimal but competitive performance. Finally, we aggregated the inference outputs of multiple models using majority voting, where ensembles combining more distinct architectures, such as TCN and Transformer, proved particularly effective. **Here we present the training code for three types of models: CNN+TCN, CLDNN, and Transformer.**

## Installation:

```python
pip install pnpl lightning transformers braindecode wandb
```

## Preparing:

Before running the training code, you need to modify the hyperparameters in **config.py**.

| Hyperparameters     |                         Description                          |
| :------------------ | :----------------------------------------------------------: |
| BASE_PATH           | Your PNPL data path. If you haven't downloaded it in advance, it will be automatically downloaded to this path. |
| SENSORS_SPEECH_MASK | Whether to apply masking operations to some sensors. The default is "slice(None)". |
| OUTPUT_DIR          |                        Output folder                         |
| LOG_DIR             |                        Log file path                         |
| CHECKPOINT_PATH     |          The path for saving training checkpoints.           |
| MODEL_DIM           | The dimension of the model's intermediate features. The default value is 100, but should be changed to 128 when using transformer-based models. |
| WARM_UP             | Whether to use a learning rate warm-up scheduler. The default is "False", but should be changed to "True" when using transformer-based models. |
| TMIN, TMAX          | The left and right critical values for time window sampling. These define the starting position and length of the MEG window. |
| LABEL_WINDOW        | The range of prediction frames within the window. By default, it predicts all frames in the entire window. The numerical value represents N frames before and after the center frame of the window. |
| STRIDE              | The step size for window sliding during MEG signal sampling. |

## Training:

The default training mode is multi-GPU training, automatically managed by lightning.

You can uncomment lines 343-345 in **model.py** to select different models. SpeechModel represents the standard CLDNN model. In our team's experiments, we conducted multi-scale experiments on different strides of TCN by adjusting the window sampling stride and threshold.

```python
python train.py
```

## Evaluating:

Evaluation will be performed on the test set, calculating the corresponding Macro F1 score. Please remember to modify the corresponding model path.

```python
python evaluate.py
```

## Submission:

Perform inference on the holdout dataset and generate the corresponding CSV file. Please remember to modify the corresponding model path.

```python
python submission.py
```

For multiple CSV files voting, we use a majority voting strategy. In addition, we also trained Crisscross and SEANet models to serve as voting files. The specific CSV files are located in the **best_vote_files** folder.

```python
python vote_submission.py
```

