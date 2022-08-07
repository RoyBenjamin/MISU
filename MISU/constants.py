import os

PROJECT_DIR_PATH = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PROJECT_DIR_PATH)

CACHE_PATH = os.path.join(PROJECT_ROOT, 'cache')
CHECKPOINTS_PATH = os.path.join(PROJECT_ROOT, 'MISU', 'checkpoints')
DATA_PATH_MOLECULENET = os.path.join(PROJECT_DIR_PATH, 'data', 'molecule_net')
DATA_PATH_PCQ = os.path.join(PROJECT_DIR_PATH, 'data', 'pcq')
LOG_PATH = os.path.join(PROJECT_ROOT, 'logdir')
SINGLE_MLP = False

if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)
if not os.path.isdir(LOG_PATH):
    os.mkdir(LOG_PATH)
if not os.path.isdir(DATA_PATH_MOLECULENET):
    os.mkdir(DATA_PATH_MOLECULENET)
if not os.path.isdir(CHECKPOINTS_PATH):
    os.mkdir(CHECKPOINTS_PATH)
