#!/bin/sh

notebook=$1

if test -z "$notebook"
then
    echo 'Please specify a notebook to update.' >&2
    exit 1
fi

jupytext --to notebook "$notebook.py"
jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=None "$notebook.ipynb"
