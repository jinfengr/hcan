#!/bin/bash

for dataset in "TrecQA" "Quora" "TwitterURL"
    do
    echo ">>> Build IDF weights for ${dataset}"
    python -u generate_idf.py -d ${dataset}
done
