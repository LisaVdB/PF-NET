'''
this code performs prediction on the validation data.
'''

import numpy as np
from checkingModel import create_checkingModel

def main(pfam_labels, integer_labels):
    
    #load sequences and labels
    val_sequences_file = np.load('D:/Post-doc/Project_SoybeanPhospho/Orthologs/PF-NET/Data/validation/validation-encoded-sequences-soybean.npz')
    val_sequences = val_sequences_file['arr_0']
    val_pfam_labels = []
    top_labels = []

    '''
    check sequence-data and labels shape 
    '''
    print("-----Sequences Shape-----")
    print(val_sequences.shape)

    '''
    load model
    '''
    
    checkingModel = create_checkingModel(filters = 320)
    print('\x1b[2K\tModel created')
    
    '''
    check if model weights are saved and load weights
    load latest weights
    '''
    
    checkingModel.load_weights('D:/Post-doc/Project_SoybeanPhospho/Orthologs/PF-NET/Data/model-results/pfnet.12-0.21.hdf5')
    
    print('\x1b[2K\tModel Summary')
    checkingModel.summary()

    #Get node values before softmax
    predictions = checkingModel.predict(val_sequences, verbose = 1)
    top = np.sort(predictions, axis = 1)[:,-3:]
    ind = np.argsort(predictions, axis = 1)[:,-3:]
    
    #predictions_test = np.argmax(predictions, axis = 1)
    
    #np.savetxt('D:/Post-doc/Project_SoybeanPhospho/Orthologs/PF-NET/Data/validation/validation-predictions/test-yeast-max.txt', ind, fmt = '%1.5f')
    
    #save predictions
    for i in range(len(ind)):
        for j in range(3):
            top_labels.append(pfam_labels[ind[i,j]])
        val_pfam_labels.append(top_labels)
        top_labels = []
        
    results = np.hstack([val_pfam_labels,top])
    np.savetxt('D:/Post-doc/Project_SoybeanPhospho/Orthologs/PF-NET/Data/validation/validation-predictions/pfam-testing-predictions-soybean.txt', results, fmt = '%s')

pfam_labels = []
integer_labels = []

with open('D:/Post-doc/Project_SoybeanPhospho/Orthologs/PF-NET/Data/pfam-integer-mapping.txt') as file:
    line = file.readline()
    line = file.readline()
    while line:
        line = line.split('\t')
        pfam_labels.append(line[0])
        integer_labels.append(line[2])
        line = file.readline()

main(pfam_labels, integer_labels)
    

