from hlstm_framework.data import MusicDataPipeline
from hlstm_framework.models.trainer import train_model
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


if __name__ == "__main__":
    # MusicDataPipeline().ingest()
    train_model()