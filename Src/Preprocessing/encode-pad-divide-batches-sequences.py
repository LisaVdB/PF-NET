"""
this code runs for all the saved batches in one go. no need to run seperately for all batches.
encode sequences in binary format
padded sequences such that all have same length
"""

import numpy as np
from keras.preprocessing.sequence import pad_sequences
import os

def encode(sequences, encoding_type):
    '''
    Encoding of sequences
    Three types of encoding:
    1. One-hot encoding - Done
    2. Binary encoding - Done
    3. Integer encoding - Done
    ''' 
    encoded_amino_acid = []
    encoded_sequence = []
    amino_to_integer_mapping = {'A':1,'B':22,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,
                            'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,
                            'V':18,'W':19,'Y':20,'X':21,'U':23,'J':24,'Z':25,'O':26}

    #Binary-encoding
    if encoding_type == 'binary_encoding':
        i = 0
        for sequence in sequences:
            print("Encoding in process, Sequence number:", i)
            for amino_acid in sequence:  
                encoded_binary_format = bin(amino_to_integer_mapping[amino_acid])[2:].zfill(5)
                encoded_amino_acid.append([int(binary) for binary in str(encoded_binary_format)]) 
            encoded_sequence.append(np.asarray(encoded_amino_acid))
            encoded_amino_acid = []
            i += 1

    # One-hot-encoding
    elif encoding_type == "one_hot_encoding":
        i = 0
        for sequence in sequences:
            print("Encoding in process, Sequence number:", i)
            encoded_integer_format = [amino_to_integer_mapping[amino_acid] for amino_acid in sequence]
            #print(encoded_integer_format)
            for value in encoded_integer_format:
                letter = [0 for _ in range(27)]
                letter[value] = 1
                encoded_amino_acid.append(letter)
            encoded_sequence.append(encoded_amino_acid)
            encoded_amino_acid = []
            i += 1

    # Integer-encoding
    elif encoding_type == 'integer_encoding':
        for sequence in sequences:
            encoded_integer_format = [amino_to_integer_mapping[amino_acid] for amino_acid in sequence]
            encoded_sequence.append((np.asarray(encoded_integer_format)).reshape(-1,1))

    else:
        print("Incorrect Encoding type")
        return None

    return np.asarray(encoded_sequence)

batches_path = '../../Data/extracted-pfam-total-data/extracted-pfam-data-batches/'
batches_list = sorted(os.listdir(batches_path))

for batch in batches_list:
    
    aa_sequences = np.load(batches_path + batch + '/aa-sequences.npz', allow_pickle = True)
    aa_sequences = aa_sequences['arr_0']
    encoding_type = "binary_encoding"

    # encode sequences
    encoded_sequences = encode(aa_sequences, encoding_type)
    print(encoded_sequences.shape)
    print("Encoding done!!")

    # pad sequences
    padded_sequences = pad_sequences(encoded_sequences, padding='post', maxlen = 1234)
    print("Padding done!!!")
    padded_sequences = padded_sequences.astype('int8')

    # save encoded sequences
    np.savez_compressed(batches_path + batch + '/encoded-sequences.npz', padded_sequences)
    