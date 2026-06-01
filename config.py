import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Model
    NUM_PITCHES = int(os.getenv("MODEL_NUM_PITCHES", 128))
    NUM_DURATIONS = int(os.getenv("MODEL_NUM_DURATIONS", 128))
    NUM_VELOCITIES = int(os.getenv("MODEL_NUM_VELOCITIES", 128))
    NUM_INSTRUMENTS = int(os.getenv("MODEL_NUM_INSTRUMENTS", 128))
    STYLE_CLASSES = int(os.getenv("MODEL_STYLE_CLASSES", 2))
    EMBED_DIM = int(os.getenv("MODEL_EMBED_DIM", 256))
    CONTROL_DIM = int(os.getenv("MODEL_CONTROL_DIM", 128))
    HIDDEN_DIM = int(os.getenv("MODEL_HIDDEN_DIM", 512))
    DROPOUT = float(os.getenv("MODEL_DROPOUT", 0.3))
    DEVICE = os.getenv("MODEL_DEVICE", "mps")

    # Training
    SEQ_LEN = int(os.getenv("TRAIN_SEQ_LEN", 128))
    BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", 32))
    NUM_EPOCHS = int(os.getenv("TRAIN_NUM_EPOCHS", 30))
    LR = float(os.getenv("TRAIN_LR", 0.001))
    VAL_SPLIT = float(os.getenv("TRAIN_VAL_SPLIT", 0.2))
    PATIENCE = int(os.getenv("TRAIN_PATIENCE", 7))
    TOKEN_PATH = os.getenv("TRAIN_TOKEN_PATH", "data/encoded/encoded_tokens_25_new.pkl")
    OUTPUT_DIR = os.getenv("TRAIN_OUTPUT_DIR", "models")
    LOG_DIR = os.getenv("TRAIN_LOG_DIR", "runs/hlstm")

    # Sprint 3 — Model architecture upgrades
    USE_POSITION_ENCODING = os.getenv("MODEL_USE_POSITION_ENCODING", "false").lower() == "true"
    NUM_POSITIONS = int(os.getenv("MODEL_NUM_POSITIONS", 64))
    USE_CHECKPOINTING = os.getenv("MODEL_USE_CHECKPOINTING", "false").lower() == "true"
    USE_TORCH_COMPILE = os.getenv("MODEL_USE_TORCH_COMPILE", "false").lower() == "true"

    # Sprint 4 — Deep architecture upgrades
    USE_CROSS_ATTENTION = os.getenv("MODEL_USE_CROSS_ATTENTION", "false").lower() == "true"
    USE_INSTR_ATTN_POOLING = os.getenv("MODEL_USE_INSTR_ATTN_POOLING", "false").lower() == "true"
    USE_TOKEN_FUSION = os.getenv("MODEL_USE_TOKEN_FUSION", "false").lower() == "true"

    # Sprint 1 — Training performance upgrades
    NUM_WORKERS = int(os.getenv("TRAIN_NUM_WORKERS", 2))
    PIN_MEMORY = os.getenv("TRAIN_PIN_MEMORY", "true").lower() == "true"
    USE_AMP = os.getenv("TRAIN_USE_AMP", "true").lower() == "true"
    LOSS_WEIGHTS = os.getenv("TRAIN_LOSS_WEIGHTS", "1.0,1.0,1.0,1.0")  # pitch,dur,vel,instr
    LR_SCHEDULER = os.getenv("TRAIN_LR_SCHEDULER", "plateau")  # "plateau", "cosine", or "none"
    LR_PATIENCE = int(os.getenv("TRAIN_LR_PATIENCE", 3))
    LR_MIN = float(os.getenv("TRAIN_LR_MIN", 1e-6))
    GRAD_CLIP_NORM = float(os.getenv("TRAIN_GRAD_CLIP_NORM", 0.0))  # 0 = disabled

    # Preprocessing
    DATA_DIR = os.getenv("PREPROCESS_DATA_DIR", "/Users/kathi.s/ML_Projects/thesis/SymphonyNet")
    ENCODED_PATH = os.getenv("PREPROCESS_ENCODED_PATH", "data/encoded/encoded_tokens_25_new.pkl")
    SUB_DIRS = os.getenv("PREPROCESS_SUB_DIRS", "classical,contemporary").split(",")
    MIN_FILES = int(os.getenv("PREPROCESS_MIN_FILES", 25))

    # Experimentation
    INPUT_FILE = os.getenv("EXP_INPUT_FILE", "test/20centuryfox.mid")
    PRESET = os.getenv("EXP_PRESET", "gentle")
    NUM_SAMPLES = int(os.getenv("EXP_NUM_SAMPLES", 3))
    STYLE = int(os.getenv("EXP_STYLE", 1))
    INSTRUMENTS = [int(i) for i in os.getenv("EXP_INSTRUMENTS", "40,41,42,43").split(",")]
