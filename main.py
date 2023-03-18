import os
import sys
sys.path.append(os.path.dirname(os.path.abspath('.')))

from dataset_utils import *
from utils import *

# Load affwild data
aw_train_df, aw_val_df = loadAffwild2()

from utils import adapt_and_test_on_affwild2
from models import FEClassifier
import functools
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

arch = 'seresnext101'
model_file_name = f'{arch}_trained'
proto_model = functools.partial(FEClassifier, arch, [128])
state_dict = torch.load(model_file_name, map_location=torch.device('cpu'))

results_df_after = adapt_and_test_on_affwild2(proto_model,
                                              state_dict,
                                              aw_val_df,
                                              device,
                                              momentum=0.01,
                                              max_steps=3,
                                              lr=1e-4,
                                              method='temporal_smoothness_bn_freeze_running_stats',
                                              chunks=True,
                                              lpf=True,
                                              batch_size=160,
                                              window_size=7,
                                              on_logits=True,
                                              entropy_multiplier=True)


display(results_df_after.loc['overall'])