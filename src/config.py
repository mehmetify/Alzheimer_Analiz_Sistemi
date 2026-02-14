import os

# Proje Ana Dizini
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Veri Dizinleri
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_IMG_DIRS = [os.path.join(DATA_DIR, "raw", "OAS2_RAW_PART1")]
TEST_SAMPLE_DIR = os.path.join(DATA_DIR, "test_sample")
METADATA_PATH = os.path.join(DATA_DIR, "metadata", "oasis_data.csv")

# Görüntü İşleme Ayarları
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_DEPTH = 32
N_CHANNELS = 1

# Model Eğitim Ayarları
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.001