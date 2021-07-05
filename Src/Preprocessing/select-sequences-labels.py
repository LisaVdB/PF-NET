'''
select sequences whose length is less than the threshold length
'''
import numpy as np
import json

def select_sequences(aa_sequences, pfam_labels, crop_length):
    selected_aa_sequences = []
    selected_pfam_labels = []
    
    for sequence_index in range(len(aa_sequences)):

        assert len(aa_sequences[sequence_index]) != 1, "Invalid Sequence"
        
        if len(aa_sequences[sequence_index]) <= crop_length:
            print("Select Sequence number:", sequence_index)
            selected_aa_sequences.append(aa_sequences[sequence_index])
            selected_pfam_labels.append(pfam_labels[sequence_index])
        else:
            print("Ignore Sequence number:", sequence_index)
         


     np.savez_compressed('../../Data/extracted-pfam-total-data/aa-sequences-crop.npz', selected_aa_sequences)
     np.savez_compressed('../../Data/extracted-pfam-total-data/pfam-labels-crop.npz', selected_pfam_labels)

    return selected_aa_sequences

with open("../../Data/extracted-pfam-total-data/aa-sequences.json","r") as file:
    aa_sequences = json.load(file)

with open("../Data/extracted-pfam-total-data/pfam-labels.json","r") as file:
    pfam_labels = json.load(file)

crop_length = 1234

a = select_sequences(aa_sequences, pfam_labels, crop_length)