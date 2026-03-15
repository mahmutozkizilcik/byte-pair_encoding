# Byte-Pair Encoding (BPE) Tokenization

This repository contains an implementation of the **Byte-Pair Encoding (BPE)** algorithm, a widely used subword tokenization method in Natural Language Processing (NLP).

## Overview

The code contains two main functions that simulate the process of how modern language models build their vocabulary:

1. **`train_BPE`**: Parses a training corpus to iteratively find and merge the most frequently occurring adjacent character pairs based on a defined `max_merges` ceiling and a configurable `topK` fallback limit. Returns the optimal merge sequence alongside the updated vocabulary.
2. **`test_BPE`**: Takes an unseen testing corpus and strictly applies the previously learned merge rules in order. Transforms the text corpus into a sequence of vocabulary indices (Token IDs), suitable for model consumption.

## Usage

The code acts as a standalone script capable of loading initial unigrams, training from custom corpus (`train.txt`, `train1.txt`, `train2.txt`), and projecting the rules onto test clusters (`test.txt`, `test1.txt`, `test2.txt`). 

To run the predefined configurations and see the resulting mappings and ID lists:

```bash
python template.py
```
This generates configuration outputs directly containing the full breakdown and the vocabulary mappings inside files like `output.txt` or `myoutput.txt`.

## Features
- Dynamic variable tracking and string flattening
- Automated evaluation loop handling multiple configuration tests simultaneously.
- Frequency-aware character adjacency parsing.
