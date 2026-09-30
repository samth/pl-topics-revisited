#!/bin/sh
# Download the pldb bulk dump (dblp metadata CC0; abstracts remain their authors'/publishers' copyright).
mkdir -p data out && curl -sSL -o data/papers.jsonl.gz https://pldb.kirancodes.me/data/papers.jsonl.gz
