#!/bin/bash

# Install gdrive if you are on Linux
export PATH="$(pwd)":$PATH
if [ ! -f "gdrive-*" ]; then
  mv gdrive-* gdrive
  echo "Install gdrive done."
fi

cd data
if [ ! -d "word2vec" ]; then
  mkdir word2vec
fi
cd word2vec

echo ""
echo ">>>Download word2vec from Google Drive..."
gdrive download 0B7XkCwpI5KDYNlNUTTlSS21pQmM

echo ""
echo ">>>Download IDF weights for twitter..."
cd ../twitter
gdrive download 0B1EhxQ7GBJdsZTVmcFVMcDY1RWM
tar -xf collection_word_idf.json.tar
rm collection_word_idf.json.tar
gdrive download 0B1EhxQ7GBJdsbXdROGZQYzV5cFU
tar -xf collection_ngram_idf.json.tar
rm collection_ngram_idf.json.tar

echo ""
echo ">>>Build trec_eval tool..."
cd ../..
tar -xf trec_eval.8.1.tar.gz
cd trec_eval.8.1
# suppress make warnings
make --ignore-errors 2> make.log
rm make.log

echo ">>>Done."
