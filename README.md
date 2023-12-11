# Enzyme-Classification 
## Description
Given a library of labelled sequences from some well-known organisms, the task is to create a model that can label sequences from new organisms. Each sequence could represent any kind of enzyme - not just kinases.

All enzymes are made of one or more chains of amino acids, which determine their structure, behaviour, and interactions with other enzymes and molecules. That means it should be possible to predict the protein’s function and behaviour given just the amino acid sequence.

A model able to perform this task would have many applications. In addition to enzymes from known organisms (which we have from studying their proteomes), there are vast numbers of metagenomic sequences - this is proteomic sequence data from environmental samples. Being able to quickly annotate them with function using this model (i.e. going beyond simple sequence similarity) would be indispensable. Models developed in the course of this challenge may contribute to furthering the understanding of the world around us.

## Dataset
The data was provided by InstaDeep, Initially for a competition on ZINDI. It consists of labelled amino acid sequences. Each sequence has a unique ID, the amino acid sequence, the organism it came from and the label. You must predict the label for the test set. Labels consist of one of 20 classes. There are ten organisms, 8 in the training set and 2 in the test set. Sequences above a set length have been excluded from this dataset.
