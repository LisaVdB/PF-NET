# PF-NET: a neural network to predict the protein family of input sequences

PF-NET is a multi-layer neural network, consisting of a CNN, attention layer, and a biLSTM, that accurately annotates sequences from 996 protein families. 

## Publication
XX

## Requirements
- 

## Training and testing datasets
We selected 996 protein families from Pfam (https://pfam.xfam.org/), focusing on protein families within the plant and animal kingdom. We extracted accompanying sequences from Pfam's underlying sequence database using the following tables: pfama_reg_full_significant and pfamseq. A third table (pfamnn) was created that contains the 996 protein families. MySQL scripts to generate the pfamnn table and to extract sequences are located in Src/MySQL. 

The total training dataset consisted of 7,385,028 sequences, covering the entire tree of life. Sequences with a length < 1234 amino acids were kept.

## Model architecture
