from .train_and_eval import train_and_eval as train
from .inference import recognize
from .utils import assessments

__version__ = '0.8.0rc'

__all__ = ['prepare', 'train', 'recognize', 'assessments']
