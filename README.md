# Supplementary code for the paper "Quantitative and real-time analysis of neurogenesis with single cell resolution in mouse tissue reveals stochastic and periodic expression dynamics of HES5"

Dependencies (tested on Ubuntu 16.04):

- nosetests
- matplotib
- pandas
- seaborn
- numba
- PyDDE (https://github.com/hensing/PyDDE) 
- the ubuntu package python-tk (sudo apt install python-tk)

Please see the file `troubleshooting.md` for typical installation errors.

## File structure

- `/src/` contains the main Python module for this repository, `hes5.py`. It includes functions to simulate stochastic and deterministic model traces, and to calculate power spectra etc.
- `/test/` contains the code to generate simulated data in the paper, and to make the paper figures. The main modules in this folder are `test_make_final_figures.py` and `test_make_analysis_for_paper.py`
- `/test/data` contains simulated data for this paper in the form of numpy arrays

## Running the code

We use the nosetests testing infrastructure to run all code in this project. Code can be run using the nostests command like this:

~~~
cd ./test
nosetests -s test_infrastructure.py
~~~

This will run all functions in `test/test_infrastructure.py` whose name starts with `test_`, and it is possible to toggle whether individual functions are run by renaming them, for example by replacing `test_` with `xest_`. The same concept applies to all python modules in the `/test` folder. The file `test_infrastructure` contains a few functional tests for main functions in `src/hes5.py`, and the nosetests command above can be used to see whether the project is correctly set up.

For any questions concerning this repository, please email Jochen Kursawe at `jochen dot kursawe at manchester dot ac dot uk`

