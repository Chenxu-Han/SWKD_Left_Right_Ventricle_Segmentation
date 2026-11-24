from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

IMAGE_SIZE = 128
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SEED = 42
EPOCHS = 50
PATIENCE = 15
ALPHA = 0.15
GAMMA = 1.5
TEMPERATURE = 2.0
MARGIN = 1.0
EPSILON = 1e-6
DICE_WEIGHT = 1.0
UAW_WEIGHT = 1.0
DATASETS = ("Public_EchoNet_Dynamic", "Public_CAMUS", "Public_AMED", "Public_EchoCP", "Public_cardiacUDA", "ShanTouShiChaoShengYiQiYanJiuSuoGuFenYouXianGongSi", "BeiJingDaXueDiYiYiYuan",)


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
