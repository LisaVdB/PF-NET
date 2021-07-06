
import numpy as np
from model import create_model
import os




def main():
    
    #load sequences and labels
    batch_file_path = '../../Data/stratified-dataset/train-test-splits-single-label-batches/'
    batches_files_list = sorted(os.listdir(batch_file_path))
    
    model_weights_path = '../../Data/model-results/single-labels/batch-1-3-4/checkpoints/'
    
    test_sequences = []
    test_labels = []
    
    for batch in batches_files_list:
            
        test_seq_file = np.load(batch_file_path + batch + '/test-sequences.npz')
        test_sequences.extend(test_seq_file['arr_0'])

        test_labels_file = np.load(batch_file_path + batch + '/test-labels.npz')
        test_labels.extend(test_labels_file['arr_0']) 
    
    '''
    check sequence-data and labels shape 
    '''
    test_sequences = np.array(test_sequences)
    test_labels = np.array(test_labels)
    
    print("-----Sequences Shape-----")
    print(test_sequences.shape)
    
    print("-----Labels Shape-----")
    print(test_labels.shape)

    np.savez_compressed('../../Data/model-results/single-labels/all-test-labels.npz', np.argmax(test_labels, axis = 1))
    del test_labels
    '''
    load model
    '''

    model = create_model(filters = 320)
    print('\x1b[2K\tModel created')


    '''
    check if model weights are saved and load weights
    '''
    if os.path.exists(model_weights_path + 'pfnet.12-0.21.hdf5'):
        model.load_weights(model_weights_path + 'pfnet.12-0.21.hdf5')
        print('\x1b[2K\tWeights loaded')
    else:
        print('\x1b[2K\tNo weights present!!')
        


    print('\x1b[2K\tModel Summary')
    model.summary()

    predictions = model.predict(test_sequences, verbose = 1)
    predictions = np.argmax(predictions, axis = 1)

    #save predictions
    np.savez_compressed('../../Data/model-results/single-labels/all-test-predictions.npz', predictions)
    



main()

