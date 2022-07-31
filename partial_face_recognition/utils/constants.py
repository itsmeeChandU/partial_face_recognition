import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUESTS_PATHNAME_PREFIX='/'
TRAIN_IMAGES_PATH = os.path.join(ROOT, "data/train")
TEST_IMAGES_PATH = os.path.join(REQUESTS_PATHNAME_PREFIX, "data/test")

