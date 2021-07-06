import numpy as np
import pickle
from model import create_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import keras.backend as K
from sklearn.utils import class_weight
import os
from loss import *
import sys


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def main():
    loss_type = sys.argv[1]
    '''
    loading saved-data batch
    '''
    
    # change last folder name by respective batches
    save_file_path = '../../Data/model-results/single-labels/batch-1-3-4/'
    
    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)
        os.mkdir(save_file_path + 'checkpoints/')
        print('-----Directory made-----')
    else:
        print('-----Directory already exists-----')
        
        
        
    load_data_path = '../../Data/stratified-dataset/train-test-splits-single-label-batches/'
    
    # load three different combination of batches

    batch_1_path = load_data_path + 'batch-1/'
    batch_2_path = load_data_path + 'batch-3/'
    batch_3_path = load_data_path + 'batch-4/'

    num_batches = 3
    batches = [batch_1_path, batch_2_path, batch_3_path]
    
    print('Loading train-validation-test sequences and labels!')

    # load 1st batch data
    train_seq = [], train_label = [], test_seq = [], test_label = [], val_seq = [], val_label = []
    
    for i in range(num_batches):
        
        train_seq_file = np.load(batches[i] + 'train-sequences.npz')
        train_seq.extend(train_seq_file['arr_0'])
        
        train_label_file = np.load(batches[i] + 'data/train-labels.npz')
        train_label.extend(train_label_file['arr_0'])

        val_seq_file = np.load(batches[i] + '/validation-sequences.npz')
        val_seq.extend(val_seq_file['arr_0'])
    
        val_label_file = np.load(batches[i] + 'validation-labels.npz')
        val_label.extend(val_label_file['arr_0'])

        test_seq_file = np.load(batches[i] + 'test-sequences.npz')
        test_seq.extend(test_seq_file['arr_0'])
    
        test_label_file = np.load(batches[i] + 'test-labels.npz')
        test_label.extend(test_label_file['arr_0'])       
        
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    val_label = np.array(val_label)
    
    train_seq = np.array(train_seq)
    test_seq = np.array(test_seq)
    val_seq = np.array(val_seq)
    
    print('Train-validation-test sequences and labels loaded!')

    print('\x1b[2K\tTraining sequence data and labels shape')
    print(train_seq.shape)
    print(train_label.shape)

    print('\x1b[2K\tValidation sequence data and labels shape')
    print(val_seq.shape)
    print(val_label.shape)

    print('\x1b[2K\tTesting sequence data and labels shape')
    print(test_seq.shape)
    print(test_label.shape)
    
    np.savez_compressed(save_file_path + 'test-labels.npz', test_label)

   
    model = create_model(filters = 320)
    print('\x1b[2K\tModel created')

    '''
    check if model weights are saved and load weights
    '''
    if os.path.exists('../..Data/model-results/single-labels/batch-2-3-4/checkpoints/' + 'pfnet.12-0.25.hdf5'):
        model.load_weights('../..Data/model-results/single-labels/batch-2-3-4/checkpoints/' + 'pfnet.12-0.25.hdf5')
        print('Weights loaded')
    else:
        print('No weights present start from scratch!!')   

    print('\x1b[2K\tCompiling Model')

    #learning rate decay
    initial_learning_rate = 1e-5
    epochs = 12
    decay = initial_learning_rate/epochs
    print("Decay", decay)
    def lr_time_based_decay(epoch, lr):
        return lr * 1/(1 + decay * epoch)


    #loss function
    if loss_type == 'focal-loss':
        loss_function = focal_loss(alpha = 0.45)
        print('\x1b[2K\tFocal Loss is the loss function')
    else:
        loss_function = tf.keras.losses.CategoricalCrossentropy()
        print('\x1b[2K\tCategorical Cross Entropy is the loss function.')

    metric = [tf.keras.metrics.Precision(name = 'precision'),
    tf.keras.metrics.Recall(name = 'recall'), 
    tf.keras.metrics.AUC(name = 'auc', curve = 'ROC'),
    tf.keras.metrics.CategoricalAccuracy(name ='accuracy'),
    tf.keras.metrics.AUC(name = 'aupr', curve = 'PR'),
    f1_metric
    ]
    
    model.compile(loss = loss_function,\
        optimizer = tf.keras.optimizers.Adam(learning_rate = initial_learning_rate),\
             metrics = metric)
    
    print('\x1b[2K\tModel Compiled')
    print('\x1b[2K\tModel Summary')
    model.summary()

    #model checkpoints
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = save_file_path + 'checkpoints/tbinet.{epoch:02d}-{val_loss:.2f}.hdf5',\
        save_weights_only=True,\
            monitor='val_loss',\
                mode='auto',\
                    save_best_only=True)
    
    #early stopping
    earlystopper = EarlyStopping(monitor='val_loss', patience=7, verbose=1)

    #class weights
    ground_truth = np.argmax(train_label, axis=-1)
    class_weights = class_weight.compute_class_weight('balanced', classes = np.unique(ground_truth), \
        y = ground_truth)


    history = model.fit(train_seq, train_label, batch_size = 100, shuffle = True, epochs = 12,\
        verbose = 1, validation_data = (val_seq, val_label),\
            class_weight = class_weights, \
                callbacks=[model_checkpoint_callback, earlystopper, LearningRateScheduler(lr_time_based_decay)])

    results = model.evaluate(test_seq, test_label, verbose = 1)
    print(model.metrics_names)
    predictions = model.predict(test_seq, verbose = 1)

    del train_seq, train_label, val_seq, val_label, test_seq, test_label

    #save evaluation results of test set
    with open(save_file_path + 'test-set-results.json', 'wb') as file:
        pickle.dump(results, file)

    #save model history
    with open(save_file_path + 'model-history.json', 'wb') as file:
        pickle.dump(history.history, file)

    #save predictions
    np.savetxt(save_file_path + 'predictions-test.csv', predictions)

    report_file =  open(save_file_path + 'training_details.txt', "w")
    report_file.write("Learning rate: {}".format(initial_learning_rate))
    #report_file.write("\nFocal loss (alpha): {}".format(fl_alpha))
    report_file.write("\nModel metrics: {}".format(model.metrics_names))
    report_file.write("\nModel was trained on batch 1 and 4")
    report_file.close()
    
main()