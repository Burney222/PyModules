# Repository for helpful functions and classes

A short overview of the classes and functions is given:

## ValAndErr.py

Class that handles simple calculations with values and uncertainties.

## efficiency.py 

Class that handles efficiencies:
  * calc_efficiency(nEvents, nAll): Returns ValAndErr object with the efficiency and a binomial uncertainty.
  * eff(df, cut, prev_cut="", weight=""): calculates efficiencies on Pandas DataFrames with cuts. Optional weighted data can be processed and previous cuts can be given.

## utilities.py

General useful functions:
  * ensure_dir(directory): checks if directory exists and if not creates it.

## ROOT_utils.py

Useful ROOT helper functions: 
  * saveCanvas(can, name): saves canvas in pdf/png/C/root formats.

## ganga_utils.py

Ganga functions.

## Plotting

Plot with pulls, plot correlations... Description given in sub-directory.
