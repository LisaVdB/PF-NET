
import numpy as np
from model import create_model
import os




def main():
    
    i = 2
    #load sequences and labels
    load_file_path = '../../Data/model-results/single-labels/batch-1-5-6/'
    
    batch = load_file_path + 'batch-1-5-6/'
    test_sequences_file = np.load(load_file_path + 'test_sequences_1.npz')
    test_sequences = test_sequences_file['arr_0']

    test_sequences_file = np.load(load_file_path + 'test_sequences_5.npz')
    test_sequences = np.append(test_sequences, test_sequences_file['arr_0'], axis = 0)

    test_sequences_file = np.load(load_file_path + 'test_sequences_6.npz')
    test_sequences = np.append(test_sequences, test_sequences_file['arr_0'], axis = 0) 
    
    '''
    check sequence-data and labels shape 
    '''
    print("-----Sequences Shape-----")
    print(test_sequences.shape)

    '''
    load model
    '''

    model = create_model(filters = 320)
    print('\x1b[2K\tModel created')


    '''
    check if model weights are saved and load weights
    '''
    if os.path.exists(load_file_path + 'checkpoints/pfnet.12-0.21.hdf5'):
        model.load_weights(load_file_path + 'checkpoints/pfnet.12-0.21.hdf5')
        print('\x1b[2K\tWeights loaded')
    else:
        print('\x1b[2K\tNo weights present!!')
        


    print('\x1b[2K\tModel Summary')
    model.summary()

    predictions = model.predict(test_sequences, verbose = 1)
    predictions = np.argmax(predictions, axis = 1)

    #save predictions
    np.savetxt(load_file_path + 'predictions-test.csv', predictions)



main()

