# hesdynamics
How can Hes5 oscillations in spinal cord neural progenitor cells be described?

Dependencies (tested on Ubuntu 16.04):

-nosetests
-matplotib
-pandas
-seaborn
-numba
-PyDDE (https://github.com/hensing/PyDDE) 
-the ubuntu package python-tk (sudo apt install python-tk)

For me, installing PyDDE threw some errors - I fixed them by editing setup.py and changing the line 

EXTRA_COMPILE_ARGS = []

to

EXTRA_COMPILE_ARGS = ['-Wno-error=-Wformat,-Wunused_but_set_variable']

I was then able to install PyDDE by entering the directory PyDDE-master and typing

sudo python setup.py install
