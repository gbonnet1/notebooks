#!/bin/sh

jupytext --to notebook monge_ampere.py
jupyter nbconvert --to notebook --inplace --execute monge_ampere.ipynb
