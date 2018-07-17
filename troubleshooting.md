# Troubleshooting

For me, installing PyDDE threw some errors - I fixed them by editing setup.py and changing the line 

`EXTRA_COMPILE_ARGS = []`

to

`EXTRA_COMPILE_ARGS = ['-Wno-error=-Wformat,-Wunused_but_set_variable']`

I was then able to install PyDDE by entering the directory PyDDE-master and typing

`sudo python setup.py install`

In order to get git cloning to work with HTTP on the manchester cluster I need to follow these instructions:

https://www.a2hosting.co.uk/kb/developer-corner/version-control-systems1/403-forbidden-error-message-when-you-try-to-push-to-a-github-repository

Further, on the manchester research cluster tests need to be invoked using commands like this:

~~~
CONDA_PREFIX=/opt/gridware/apps/binapps/anaconda/3/4.2.0
LD_PRELOAD=$CONDA_PREFIX/lib/libmkl_core.so:$CONDA_PREFIX/lib/libmkl_sequential.so nosetests -s test_my_test.py
~~~
