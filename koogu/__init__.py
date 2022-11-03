from .data import preprocess as prepare     # legacy name; remove it some day
from .train_and_eval import train_and_eval as train
from .inference import recognize
from .utils import assessments

__version__ = '0.7.1'
