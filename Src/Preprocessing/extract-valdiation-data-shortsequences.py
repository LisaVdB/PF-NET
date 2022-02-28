# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from Bio import SeqIO
import numpy as np
from keras.preprocessing.sequence import pad_sequences

#kinases yeast
# newpredict = np.array(["YGL021W","YDR059C","YPR091C","YJL136W-A","YOR119C","YOR293C-A",
#                        "YNL033W","YLR256W","YNL019C","YBL113C","YHR213W-B","YDR450W",
#                        "YML026C","YBR007C","YKL025C","YDR017C","YAR064W","YBR203W",
#                        "YPL152W-A","YOL149W","YGL258W-A"])

#phosphatases yeast
newpredict = np.array(["YOR072W-B","YNL032W","YNL128W","YBR044C","YOR137C","YNL099C",
                       "YMR283C","YPL030W","YNL036W","YHR012W","YLR048W","YDR067C",
                       "YHR067W","YOR179C","YNL111C","YOR110W","YGL220W","YOL110W",
                       "YPL062W","YOR173W","YEL035C","YDR051C","YML081W"])

data = list(SeqIO.parse("D:/Google_Drive/Post-doc/Project_SoybeanPhospho/Orthologs/PF-NET/Data/validation/Yeast_S288C_orf_trans_R64-3-1_20210421.fasta", "fasta"))

sequences = []
sequences_org = []
label = []

for index in range(len(data)):
    if data[index].id in newpredict:
        sequences.append(data[index].seq)
        sequences_org.append(data[index].seq)
        label.append(data[index].id)

threshold_length = 100 #needs to be even number
broken_sequences = []
broken_labels = []
sequences_list = []
labels_list = []
    
for index in range(len(sequences)):
    if len(sequences[index]) > threshold_length:
        while len(sequences[index]) > threshold_length:
            broken_sequences.append(sequences[index][:threshold_length])
            sequences[index] = sequences[index][threshold_length:]
        if sequences[index] is not None:
            broken_sequences.append(sequences[index])
        
        length_of_broken_seq = len(broken_sequences)
        broken_labels = [label[index]] * length_of_broken_seq    

        sequences_list.extend(broken_sequences)
        labels_list.extend(broken_labels)
        broken_sequences = []
        broken_labels = []
        
    else:
       sequences_list.append(sequences[index])
       labels_list.append(label[index])
 
           
encoded_amino_acid = []
encoded_sequence = []
amino_to_integer_mapping = {'A':1,'B':22,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,
                            'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,
                            'V':18,'W':19,'Y':20,'X':21,'U':23,'J':24,'Z':25,'O':26}

for sequence in sequences_list:
        for amino_acid in sequence:  
            if amino_acid not in amino_to_integer_mapping.keys():
                continue
            else:
                encoded_binary_format = bin(amino_to_integer_mapping[amino_acid])[2:].zfill(5)
                encoded_amino_acid.append([int(binary) for binary in str(encoded_binary_format)]) 
        encoded_sequence.append(np.asarray(encoded_amino_acid))
        encoded_amino_acid = []

mlen = int(((1234-threshold_length)/2)+threshold_length)
padded_sequences = pad_sequences(np.asarray(encoded_sequence), maxlen = mlen, padding='pre')
padded_sequences = pad_sequences(np.asarray(padded_sequences), maxlen = 1234, padding='post')
print("Padding done!!!")
padded_sequences = padded_sequences.astype('int8')
np.savez_compressed('D:/Google_Drive/Post-doc/Project_SoybeanPhospho/Orthologs/PF-NET/Data/validation/validation-encoded-sequences-yeast-newpredicted.npz', padded_sequences)
np.savetxt('D:/Google_Drive/Post-doc/Project_SoybeanPhospho/Orthologs/PF-NET/Data/validation/validation-labels-yeast-newpredicted.txt', labels_list, fmt='%s')

          


