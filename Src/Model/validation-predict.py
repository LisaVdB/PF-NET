'''
this code performs prediction on the validation data.
'''

import numpy as np
from model import create_model

def main(pfam_labels, integer_labels):
    
    #load sequences and labels
    val_sequences_file = np.load('../../Data/testing/testing-encoded-sequences.npz')
    val_sequences = val_sequences_file['arr_0']
    val_pfam_labels = []

    '''
    check sequence-data and labels shape 
    '''
    print("-----Sequences Shape-----")
    print(val_sequences.shape)

    '''
    load model
    '''

    model = create_model(filters = 320)
    print('\x1b[2K\tModel created')


    '''
    check if model weights are saved and load weights
    load latest weights
    '''
    
    model.load_weights('../../Data/model-results/single-labels/batch-1-3-4/checkpoints/pfnet.12-0.21.hdf5')
        
    print('\x1b[2K\tModel Summary')
    model.summary()

    predictions = model.predict(val_sequences, verbose = 1)
    predictions = np.argmax(predictions, axis = 1)

    #save predictions
    np.savetxt('../../Data/testing/integer-val-predictions.txt', predictions, fmt = '%d')

    for index in range(len(predictions)):
        val_pfam_labels.append(pfam_labels[predictions[index]])
        
    np.savetxt('../../Data/testing/pfam-testing-predictions.txt', val_pfam_labels, fmt = '%s')
    
pfam_labels = []
integer_labels = []

with open('../../Data/pfam-integer-mapping.txt') as file:
    line = file.readline()
    line = file.readline()
    while line:
        line = line.split('\t')
        pfam_labels.append(line[0])
        integer_labels.append(line[2])
        line = file.readline()

main(pfam_labels, integer_labels)
    

