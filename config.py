# config.py

BASE_PATH = "/home/gongzx/links/scratch/LibriBrain/data"
# BASE_PATH = "pnpl/LibriBrain"
#SENSORS_SPEECH_MASK = [18, 20, 22, 23, 45, 120, 138, 140, 142, 143, 145,
#                        146, 147, 149, 175, 176, 177, 179, 180, 198, 271, 272, 275]
SENSORS_SPEECH_MASK = slice(None)
# OUTPUT_DIR = "/cpfs04/user/lidongyang/workspace/PNPL_competition/outputs"
OUTPUT_DIR = "/home/gongzx/links/scratch/LibriBrain/data/outputs"

LOG_DIR = f"{OUTPUT_DIR}/lightning_logs"
CHECKPOINT_PATH = f"{OUTPUT_DIR}/models/speech_model.ckpt"
RESULTS_DIR = f"{OUTPUT_DIR}/results"


# Hyperparameters
INPUT_DIM = 306
MODEL_DIM = 100
WARM_UP = False
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3
LSTM_LAYERS = 2
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32
BATCH_NORM = False
BI_DIRECTIONAL = True
SMOOTHING = 0.1
POS_WEIGHT = 1.0
NUM_WORKERS = 12
TMIN = 0.0
TMAX = 8.0
LABEL_WINDOW = int(125 * (TMAX - TMIN))
STRIDE = 30

SAMPLES = 240183

