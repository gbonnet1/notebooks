#!/bin/sh

jupytext --to notebook monge_ampere.py
jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=None monge_ampere.ipynb
