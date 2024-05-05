from kub_interface import *
from opt_utils  import *
from kub_config import *

run_dir = os.path.join(pool_dir, 'running')
new_dir = os.path.join(pool_dir, 'new')
completed_dir = os.path.join(pool_dir, 'completed')

retrieve_result(1)