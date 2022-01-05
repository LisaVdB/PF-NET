'''
this code extracts validation sequences and labels from the provided dataset
encoding of sequences is also performed.
'''

from Bio import SeqIO
import numpy as np
from keras.preprocessing.sequence import pad_sequences
global count


data = list(SeqIO.parse("D:/Google_Drive/Post-doc/Project_SoybeanPhospho/Orthologs/PF-NET/Data/validation/Yeast_S288C_orf_trans_R64-3-1_20210421.fasta", "fasta"))

val_aa_sequences = []
val_labels = []


for index in range(len(data)):
    val_labels.append(data[index].id)
    val_aa_sequences.append(data[index].seq)


def break_sequences(sequence, label): 
    threshold_length = 1234
    broken_sequences = []
    broken_labels = []
    
    
    while len(sequence) > threshold_length:
        print('Big sequence!')
        broken_sequences.append(sequence[:1234])
        sequence = sequence[1234:]
        
    if sequence is not None:
        broken_sequences.append(sequence)
        
    length_of_broken_seq = len(broken_sequences)
    
    broken_labels = [label] * length_of_broken_seq    
        
    return broken_sequences, broken_labels

def convert_and_select_sequences(val_aa_sequences, crop_length, val_labels, count):
    selected_val_sequences = []
    selected_val_labels = []
    
    broken_val_sequences_list = []
    broken_val_labels_list = []
    
    for index in range(len(val_aa_sequences)):
        if len(val_aa_sequences[index]) > crop_length:
            count += np.ceil(len(val_aa_sequences[index])/1234)
            broken_sequences, broken_labels = break_sequences(val_aa_sequences[index], val_labels[index])
            broken_val_sequences_list.extend(broken_sequences)
            broken_val_labels_list.extend(broken_labels)
        else:
            selected_val_sequences.append(val_aa_sequences[index])
            selected_val_labels.append(val_labels[index])
            

    return selected_val_sequences, selected_val_labels, broken_val_sequences_list, broken_val_labels_list, count

def encode(sequences):
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


    i = 0
    for sequence in sequences:
        print("Encoding in process, Sequence number:", i)
        for amino_acid in sequence:  
            if amino_acid not in amino_to_integer_mapping.keys():
                continue
            else:
                encoded_binary_format = bin(amino_to_integer_mapping[amino_acid])[2:].zfill(5)
                encoded_amino_acid.append([int(binary) for binary in str(encoded_binary_format)]) 
        encoded_sequence.append(np.asarray(encoded_amino_acid))
        encoded_amino_acid = []
        i += 1

    return np.asarray(encoded_sequence)

count = 0
sel_val_seq, sel_val_label, brok_val_seq, brok_val_label, count = \
    convert_and_select_sequences(val_aa_sequences, 1234, val_labels, count)

val_aa_seq = sel_val_seq + brok_val_seq
val_lab = sel_val_label + brok_val_label

np.savez_compressed('D:/Google_Drive/Post-doc/Project_SoybeanPhospho/Orthologs/PF-NET/Data/validation/validation-amino-acid-sequences-yeast.npz', val_aa_seq)

val_enc_sequences = encode(val_aa_seq)
val_padded_sequences = pad_sequences(val_enc_sequences, maxlen = 1234, padding='post')
print("Padding done!!!")
val_padded_sequences = val_padded_sequences.astype('int8')

np.savez_compressed('D:/Google_Drive/Post-doc/Project_SoybeanPhospho/Orthologs/PF-NET/Data/validation/validation-encoded-sequences-yeast.npz', val_padded_sequences)

np.savez_compressed('D:/Google_Drive/Post-doc/Project_SoybeanPhospho/Orthologs/PF-NET/Data/validation/validation-labels-yeast.npz', val_lab)
np.savetxt('D:/Google_Drive/Post-doc/Project_SoybeanPhospho/Orthologs/PF-NET/Data/validation/validation-labels-yeast.txt', val_lab, fmt='%s')
