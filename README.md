# BREAKING ENCRYPTION USING QUANTUM COMPUTING

## Content Details :
        > 01-factoring-overview.ipynb - Jupyter Notebook
    Provided by D-Wave as Examples on open Jupyter Server.

        > helpers (directory)
    Provided by D-Wave, contains different functions which are necessary.

        > factoring.py
    First derivative of D-Wave code which factorizes a number one time.

        > factoring-data.py
    Major Changes made, for running Code under different parameters.

        > test.py
    Simple Python Program which factorizes a number _n_ times using a _for_ loop.

        > graphs/dff.xlsx - Excel Sheet
    Contains derivative data set from original data, to plot different graphs.

        > view_graph.ipynb
    Python Program which uses given excel sheet to plot graphs.

## INSTRUCTIONS :

1. Clone the repository
2. Make sure you have latest version of python3 and pip3
3. Install dwave sdk using _pip_
    > pip install dwave-ocean-sdk
4. Configure dwave sdk, make sure you have a D-Wave Leap account before.
    > dwave config create
5. Run _factoring.py_
    > ./factoring.py
6. Compare your results with Classical Factorization
    > ./test.py

## Links:
[D-Wave Leap](https://docs.dwavesys.com/docs/latest/leap.html)
[Original Data Sheet with Screenshots](https://docs.google.com/spreadsheets/d/1Mnlzq9NqNktos9Lj9Ny19ssbGW4tBqRpLcMn9fNjHfo/edit?usp=sharing)