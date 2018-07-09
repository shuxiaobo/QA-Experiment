from models.model_data_pairs import models_in_datasets
from .attention_over_attention_reader import AoAReader
from .attention_sum_reader import AttentionSumReader
from .simple import Simple_model
from .simplerl import Simple_modelrl
from .simple1 import Simple_model1
from .simple_squad import SimpleModelSQuad
from .BiDAF import BiDAF
from .simple_squad3 import SimpleModelSQuad3
from .simple_squad4 import SimpleModelSQuad4
from .simple_squad_bidaf import SimpleModelSQuadBiDAF
from .simple_squad5 import SimpleModelSQuad5

__all__ = list(set([model for models in models_in_datasets.values() for model in models]))
