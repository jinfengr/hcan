#!/bin/bash

for dataset in "TrecQA" "Quora" "Twitter-URL"
    python -u generate_idf.py -d ${dataset}
done