import numpy as np
from sklearn.model_selection import train_test_split
import os


save_file_path = '../../Data/stratified-dataset/train-test-splits-single-label-batches/'
dest_file_list = sorted(os.listdir(save_file_path))

stratified_single_label_path = '../../Data/stratified-dataset/single-label-batches/'
stratified_single_label_list = sorted(os.listdir(stratified_single_label_path))

batch_num = 1
for i in range(0, len(stratified_single_label_list)):
    print('Loading single label sequences!')

    seq_file = np.load(stratified_single_label_path + stratified_single_label_list[i] + '/sequences/enc-sequences.npz')
    sequences = seq_file['arr_0']
    
    label_file = np.load(stratified_single_label_path + stratified_single_label_list[i] + '/labels/enc-labels.npz')
    labels = label_file['arr_0']

    print("-----Sequences and labels shape-----")
    print(sequences.shape)
    print(labels.shape)


    train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size = 0.2, stratify = labels)
    train_sequences, validation_sequences, train_labels, validation_labels = train_test_split(train_sequences, train_labels, test_size = 0.2, stratify = train_labels)

    print('\x1b[2K\tTraining sequence data and labels shape')
    print(train_sequences.shape)
    print(train_labels.shape)

    print('\x1b[2K\tValidation sequence data and labels shape')
    print(validation_sequences.shape)
    print(validation_labels.shape)

    print('\x1b[2K\tTesting sequence data and labels shape')
    print(test_sequences.shape)
    print(test_labels.shape)


    print('Saving train-validation-test sequences and labels')
    if not os.path.exists(save_file_path + 'batch-{}/'.format(batch_num)):
         os.mkdir(save_file_path + 'batch-{}/'.format(batch_num))

    np.savez_compressed(save_file_path + 'batch-{}/train-sequences.npz'.format(batch_num), train_sequences)
    np.savez_compressed(save_file_path + 'batch-{}/train-labels.npz'.format(batch_num), train_labels)

    np.savez_compressed(save_file_path + 'batch-{}/validation-sequences.npz'.format(batch_num), validation_sequences)
    np.savez_compressed(save_file_path + 'batch-{}/validation-labels.npz'.format(batch_num), validation_labels)
    
    np.savez_compressed(save_file_path + 'batch-{}/test-sequences.npz'.format(batch_num), test_sequences)
    np.savez_compressed(save_file_path + 'batch-{}/test-labels.npz'.format(batch_num), test_labels)
    batch_num += 1

    print('-----Train-validation-test sequences and labels saved-----')
