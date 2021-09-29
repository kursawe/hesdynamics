# Welcome to the Hesdynamics repository

This repository can be used to simulate gene expression dynamics of self-repressing genes. It includes optimized code for deterministic and stochastic delay differential equations to describe gene expression oscillations, as well as code for parameter inference on these equations. Parameter inference can be conducted on single-cell time series data on protein expression through the use of a delay-adapted non-linear Kalman filter, or likelihood free through Approximate Bayesian Computation. Further model analysis methods hosted here include bifurcation and power spectrum calculations.

This repository contains supplementary material for the papers
* Quantitative single-cell live imaging links HES5 dynamics with cell-state and fate in murine neurogenesis, 
  C. S. Manning, V. Biga, J. Boyd, J. Kursawe, B. Ymisson1, D. G. Spiller, C. M. Sanderson, T. Galla, M. Rattray, N. Papalopulu
  <https://doi.org/10.1038/s41467-019-10734-8>
* miR-9 mediated noise optimization of the her6 oscillator is needed for cell state progression in the Zebrafish hindbrain,
  X. Soto, V. Biga, J. Kursawe, R. Lea, P. Doostdar, N. Papalopulu
  <https://doi.org/10.1101/608604>
* Inferring kinetic parameters of oscillatory gene regulation from single cell time series data,
  J. Burton, C. S. Manning, M. Rattray, N. Papalopulu, J. Kursawe
  <https://royalsocietypublishing.org/doi/10.1098/rsif.2021.0393>
 
All computational results in these papers can be re-generated from here.

## File structure

- `/src/` contains the main Python module for this repository, `hes5.py`. It includes functions to simulate stochastic and deterministic model traces, to calculate summary statistics of expression and power spectra etc.
- `/test/` contains the code to generate simulated data in the papers, and to make the paper figures. The code that was used in the papers by Manning et al. and Soto et al. is saved in the folders `test_manning_et_al_2019` and `test_soto_et_al_2019`, respectively. Each of these paper-specific folders contains the following files
    - `test_make_final_figures.py`, which can be used to regenerate all figures in the paper from data saved in this repository and 
    - `test_make_analysis_for_paper.py`, which can be used to generate all data needed in `test_make_final_figures.py` from scratch by re-running all simulations.
    - `data/` is a folder that contains simulated data for each paper in the form of Python numpy arrays. In case of the paper by Soto et al., this folder is empty
     In case of the paper by Manning et al., users will need to download this folder from
     <https://www.dropbox.com/sh/r7taosaoy821l1x/AABQHAWfX-uNGbqMRkAFw-r6a?dl=0>.
     This is to overcome data storage limits imposed by Github.
    - All remaining files in these folders contain additional analysis that was conducted while working towards the paper.
- The file `test_infrastructure.py` contains functional tests for the most important functions in `/src/hes5.py`, and can be used to test whether the installation is working.

## Dependencies (tested on Ubuntu 16.04):

- nosetests
- matplotib
- pandas
- seaborn
- numba
- PyDDE (https://github.com/hensing/PyDDE) 
- the ubuntu package python-tk (sudo apt install python-tk)

Please see the file `troubleshooting.md` for known installation errors.

## Running the code

We use the nosetests testing infrastructure to run all code in this project. Code can be run using the nostests command like this:

~~~
cd ./test
nosetests -s test_infrastructure.py
~~~

This will run all functions in `test/test_infrastructure.py` whose name starts with `test_`, and it is possible to toggle whether individual functions are run by renaming them, for example by replacing `test_` with `xest_`. The same concept applies to all python modules in the `/test` folder. The file `test_infrastructure` contains a few functional tests for main functions in `src/hes5.py`, and the nosetests command above can be used to see whether the project is correctly set up.

Before running any tests in the subfolders `test/test_soto_et_al_2019` and `test/test_manning_et_al_2019`, we advice to first download the folders `/test/test_soto_et_al_2019/data` and `/test/test_manning_et_al_2019/data` using the links
<https://www.dropbox.com/sh/5k8wofjk97uu5ux/AADhsd41JVzIL_KJTnnj5ysCa?dl=0> and
<https://www.dropbox.com/sh/r7taosaoy821l1x/AABQHAWfX-uNGbqMRkAFw-r6a?dl=0>, respectively.


For any questions concerning this repository, please email Jochen Kursawe at `jochen dot kursawe at st-andrews dot ac dot uk`

