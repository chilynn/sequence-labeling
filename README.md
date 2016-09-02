# Sequential Labeling

- HMM
- BILSTM-CRF

# Input Format
The first column is the char, the second column is the label(BMEO), there is an empty line between two sentences
>N  B
>
>B	M
>
>A	E
>
>D	O
>
>an empty line
>
>Z	O
>
>Z	O
>
>Z	O
>
>Z	O
>
>Z	O

# Output Format
>NBAD\<@\>NBA
>
>ZZZZZ\<@\>

# Train
python train.py train.in model -v validation.in -c char_emb -e 10 -g 2
- train.in, the path of the train file
- model, the path of the saved model
- v, the path of the validation file(optional, otherwise split the train set into train and val)
- c, the char embedding file
- e, the number of epoch(optional, default 100)
- g, the id of gpu(optional, default 0)


# Test
python test.py model test.in test.out -g 2
- model, the path of model file
- test.in, the path of test file
- test.out, the path of predict file of test
- g, the id of gpu(optional, default 0)

# Embedding
the first line of the embedding file is the number of char and embedding dimension, seperating by space, e.g 5 10
the remaining line is the char and embedding vector, seperating by space, e.g N dim1 ... dim 10

# Installation Dependencies
- python 2.7
- tensorflow 0.8
- numpy
- pandas

# References
- https://github.com/manubharghav/NER
- https://github.com/glample/tagger
