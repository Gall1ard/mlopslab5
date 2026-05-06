import os
import sys
sys.path.append(os.getcwd())
from src.model_scripts.train_model import train_model

if __name__ == "__main__":
    train_model()