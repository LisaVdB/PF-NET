import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, multiply, Permute, RepeatVector
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed
from tensorflow.keras import backend as K
from tensorflow.keras.activations import relu
from tensorflow.keras.regularizers import l2

def create_model(filters = 1):

    #input layer
    sequence_input = Input(shape=(1234,5))

    # Convolutional Layer
    # output = Conv1D(filters,kernel_size=7,padding="valid",activation="relu")(sequence_input)             # kernel_size: 5 - 15, try to change number of filters (kernel 26 to 7, filter 320 to 160)
    # output = MaxPooling1D(pool_size=14, strides=14)(output)                                           # maxpooling: 5 - 15
    # output = Dropout(0.3)(output)


    output = Conv1D(filters,kernel_size=7,padding="valid",activation=None, use_bias=None, kernel_regularizer=l2(5e-4))(sequence_input) # kernel_size: 5 - 15, try to change number of filters (kernel 26 to 7, filter 320 to 160)
    output = BatchNormalization()(output)
    output = relu(output)
    output = MaxPooling1D(pool_size=14, strides = 14)(output)                                           # maxpooling: 5 - 15
    output = Dropout(0.2)(output)


    #Attention Layer
    attention = Dense(1)(output)
    attention = Permute((2, 1))(attention)
    attention = Activation('softmax')(attention)
    attention = Permute((2, 1))(attention)
    attention = Lambda(lambda x: K.mean(x, axis=2), name='attention',output_shape=(output.shape[1],))(attention)
    attention = RepeatVector(filters)(attention)
    attention = Permute((2,1))(attention)
    output = multiply([output, attention])

    #Bi-LSTM Layer
    output = Bidirectional(LSTM(filters,return_sequences=True, kernel_regularizer=l2(1e-6)))(output)
    output = Dropout(0.5)(output)                                                                     # optimize

    flat_output = Flatten()(output)

    #FC Layer
    FC_output = Dense(1024, kernel_regularizer=l2(0.001))(flat_output)                                                               # optimize
    FC_output = Activation('relu')(FC_output)

    #Output Layer
    output = Dense(996)(FC_output)
    output = Activation('softmax')(output)

    model = Model(inputs=sequence_input, outputs=output)
    print("Model is built with ", filters ," filters.")

    return model

 