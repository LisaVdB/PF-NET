'''
Divide the sequences into batches of 1000000 for easy encoding, padding, and 
training of model
'''

import numpy as np
import os

aa_sequences_file = np.load('../../Data/extracted-pfam-total-data/aa-sequences.npz', allow_pickle=True)
aa_sequences = aa_sequences_file['arr_0']

pfam_labels_file = np.load('../../Data/extracted-pfam-total-data/pfam-labels.npz')
pfam_labels = pfam_labels_file['arr_0']

chunk_size = 1000000

batches_path = '../../Data/pfam-total-data/extracted-pfam-total-data/extracted-pfam-data-batches/'
batch_num = 1
for batch in range(0, len(aa_sequences), chunk_size):
    
    if not os.path.exists(batches_path + 'batch-{}/'.format(batch_num)):
        os.mkdir(batches_path + 'batch-{}/'.format(batch_num))
        print('Directory created')
    np.savez_compressed(batches_path + 'batch-{}/aa-sequences.npz'.format(batch_num), aa_sequences[batch:batch + chunk_size])
    np.savez_compressed(batches_path + 'batch-{}/pfam-labels.npz'.format(batch_num), pfam_labels[batch:batch + chunk_size])
    batch_num += 1

