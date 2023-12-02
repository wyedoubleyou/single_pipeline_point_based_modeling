

import subprocess as sp

# smplx_cmd ='conda activate py38'
smplx_cmd ='python -m transfer_model --exp-cfg config_files/smplx2smpl.yaml'
cmd = sp.getoutput(smplx_cmd)

