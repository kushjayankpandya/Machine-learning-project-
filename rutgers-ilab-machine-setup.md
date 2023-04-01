# To execute machine learning / deep learning project in Rutgers ilab (virtual machines) follow below steps....

## Steps

* Login to Rutgers CS website to execute resources using https://weblogin.cs.rutgers.edu/guacamole-1.4.0/#/

* See details of Rutgers ilab (virtual machines) on https://report.cs.rutgers.edu/nagiosnotes/iLab-machines.html

* Login to required machine using the link mentioned in step 1.

* Utilize GPU on Rutgers ilab machines (NODE_NAME: to execute on specific node, GPU_NO: requests x GPUs)
    > srun -w <NODE_NAME> -G <GPU_NO> --pty /bin/bash

* Setup for Python specific environment
    1. Set a path to python environment
        > export PATH="$PATH:/koko/system/anaconda/bin"
    
    2. Create a new python3.9 environment
        > conda create --name mypython39  python=3.9

    3. Activate specific environment
        > source activate mypython39

    4. Install required packages
        > conda install tensorflow
        > conda install tensorflow-gpu
        > conda install torch

    5. To start in Jupyter Notebook
        > jupyter notebook --ip=`hostname`

    6. SSH from local machine to Rutgers ilab machines
        > ssh <net_id>@<machine_name>.cs.rutgers.edu