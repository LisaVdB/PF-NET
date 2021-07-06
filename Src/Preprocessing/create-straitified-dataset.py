# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:20:49 2021

@author: HP
"""
import numpy as np
import os

'''
1. extract sequences with only one label per sequence
'''

# =============================================================================
# batches_file_path = '../../Data/extracted-pfam-total-data/extracted-pfam-data-batches/'
# 
# 
# batches_list = sorted(os.listdir(batches_file_path))
# 
# i = 1
# 
# for batch in batches_list:
#     enc_seq_file = np.load(batches_file_path + batch + '/encoded-sequences.npz')
#     enc_sequences = enc_seq_file['arr_0']
#     enc_sequences = np.asarray(enc_sequences)
#     print(batch, 'appended!')
#     
#     
#     enc_labels_file = np.load(batches_file_path + batch + '/encoded-labels.npz', allow_pickle= True)
#     enc_labels = enc_labels_file['arr_0']
# 
#     print(batch, 'appended!')
#     
#     
#     pfam_labels_file = np.load(batches_file_path + batch + '/pfam-labels.npz', allow_pickle= True)
#     pfam_labels = pfam_labels_file['arr_0']
#     pfam_labels = np.asarray(pfam_labels)
#     
#     print(batch, 'appended!')
#     
#     
#     print(len(pfam_labels))
#     print(len(enc_labels))
#     print(len(enc_sequences))
# 
# 
#     indices = []
#     
#     for index in range(len(pfam_labels)):
#         if len(pfam_labels[index]) != 1:
#             indices.append(index)
#             
#     enc_sequences = np.delete(enc_sequences, indices, axis = 0)
#     enc_labels = np.delete(enc_labels, indices, axis = 0)
#     pfam_labels = np.delete(pfam_labels, indices, axis = 0)
#     
#     print('Double labels deleted!')
# 
#     print(len(pfam_labels))
#     print(len(indices))
#     print(len(enc_labels))
#     print(len(enc_sequences))
#         
#     if not os.path.exists(batches_file_path + 'batch-{}/pfam-data-one-label-batch/'.format(i)):
#         os.mkdir(batches_file_path + 'batch-{}/pfam-data-one-label-batch/'.format(i))
#     np.savez_compressed(batches_file_path + 'batch-{}/pfam-data-one-label-batch/encoded-labels.npz'.format(i), enc_labels)    
#     np.savez_compressed(batches_file_path + 'batch-{}/pfam-data-one-label-batch/encoded-sequences.npz'.format(i), enc_sequences) 
#     np.savez_compressed(batches_file_path + 'batch-{}/pfam-data-one-label-batch/pfam-labels.npz'.format(i), pfam_labels) 
#     
#     i += 1
# 
# =============================================================================
'''
2. extract sequences with multiple labels per sequence
'''
# =============================================================================
# 
# batches_file_path = '../../Data/extracted-pfam-total-data/extracted-pfam-data-batches/'
# batches_list = sorted(os.listdir(batches_file_path))
# 
# i = 1
# 
# for batch in batches_list:
#     enc_seq_file = np.load(batches_file_path + batch + '/encoded-sequences.npz')
#     enc_sequences = enc_seq_file['arr_0']
#     enc_sequences = np.asarray(enc_sequences)
#     print(batch, 'appended!')
#     
#     
#     enc_labels_file = np.load(batches_file_path + batch + '/encoded-labels.npz', allow_pickle= True)
#     enc_labels = enc_labels_file['arr_0']
# 
#     print(batch, 'appended!')
#     
#     
#     pfam_labels_file = np.load(batches_file_path + batch + '/pfam-labels.npz', allow_pickle= True)
#     pfam_labels = pfam_labels_file['arr_0']
#     pfam_labels = np.asarray(pfam_labels)
#     
#     print(batch, 'appended!')
#     
#     
#     print(len(pfam_labels))
#     print(len(enc_labels))
#     print(len(enc_sequences))
# 
# 
#     indices = []
#     
#     for index in range(len(pfam_labels)):
#         if len(pfam_labels[index]) == 1:
#             indices.append(index)
#             
#     enc_sequences = np.delete(enc_sequences, indices, axis = 0)
#     enc_labels = np.delete(enc_labels, indices, axis = 0)
#     pfam_labels = np.delete(pfam_labels, indices, axis = 0)
#     
#     print('Single labels deleted!')
# 
#     print(len(pfam_labels))
#     print(len(indices))
#     print(len(enc_labels))
#     print(len(enc_sequences))
#         
# 
#     if not os.path.exists(batches_file_path + 'batch-{}/pfam-data-multiple-labels-batch/'.format(i)):
#         os.mkdir(batches_file_path + 'batch-{}/pfam-data-multiple-labels-batch/'.format(i))
#     
#     np.savez_compressed(batches_file_path + 'batch-{}/pfam-data-multiple-labels-batch/encoded-labels.npz'.format(i), enc_labels)    
#     np.savez_compressed(batches_file_path + 'batch-{}/pfam-data-multiple-labels-batch/encoded-sequences.npz'.format(i), enc_sequences) 
#     np.savez_compressed(batches_file_path + 'batch-{}/pfam-data-multiple-labels-batch/pfam-labels.npz'.format(i), pfam_labels) 
#     
#     i += 1    
#   
# =============================================================================
 


'''
3. implement stratified-k-fold with sequences with only one family
'''

# =============================================================================
# 
# pfam_batches_path = '../../Data/extracted-pfam-total-data/extracted-pfam-data-batches/'
# pfam_batches_list = sorted(os.listdir(pfam_batches_path))
# 
# classes_path = '../../Data/extracted-pfam-total-data/pfam-data-classes/'
# 
# #load pfam-integer mapping file
# pfam_families = []
# 
# 
# # extract pfam-integer mapping
# with open('../../Data/pfam-integer-mapping.txt') as file:
#     data = file.readline()
#     while len(data):
#         data = file.readline().split('\t')
#         if len(data) == 1:
#             break
#         pfam_families.append(data[0])
# #        integer_label.append(int(data[2]))
#         
# print('Integer-pfam mapping created!')
# 
# dict_counts = {}
# count_found = 0
# 
# # extract number of sequences per label has
# 
# for index in range(len(pfam_batches_list)):
#     pfam_labels_file = np.load(pfam_batches_path + pfam_batches_list[index] + '/extracted-single-label/pfam-labels.npz', allow_pickle = True)
#     pfam_labels = pfam_labels_file['arr_0']
#     
#     print(pfam_batches_list[index], 'appended!')
#     print(len(pfam_labels))
#     
#     for pfam_label in pfam_labels:
#         pfam_index = pfam_families.index(pfam_label[0])
#         if pfam_index not in dict_counts.keys():
#             dict_counts[pfam_index] = 1
#         else:
#             count_found += 1
#             dict_counts[pfam_index] += 1
#     
# del pfam_labels
# 
# dict_thresh_counts = {}
# 
# # divide total count by 6 to divide into 6 batches
# print('Threshold for each class created')
# for pfam, count in dict_counts.items():
#     dict_thresh_counts[pfam] = int(count/6)    
# 
# 
# 
# for class_num in range(996):
#     class_seq = []
#     class_labels = []
#     for file_index in range(len(pfam_batches_list)):
#         indices = []
#         enc_seq_file = np.load(pfam_batches_path + pfam_batches_list[file_index] + '/extracted-single-label/encoded-sequences.npz')
#         enc_sequences = enc_seq_file['arr_0']
#         print(pfam_batches_list[file_index], 'appended!')
#         print(len(enc_sequences))
#     
#         enc_labels_file = np.load(pfam_batches_path + pfam_batches_list[file_index] + '/extracted-single-label/encoded-labels.npz', allow_pickle= True)
#         enc_labels = enc_labels_file['arr_0']
#         print(pfam_batches_list[file_index], 'appended!')
#         print(len(enc_labels))
#     
#         pfam_labels_file = np.load(pfam_batches_path + pfam_batches_list[file_index] + '/extracted-single-label/pfam-labels.npz', allow_pickle = True)
#         pfam_labels = pfam_labels_file['arr_0']
#         print(pfam_batches_list[file_index], 'appended!')
#         print(len(pfam_labels))
#     
#     
#         for i in range(len(enc_sequences)):
#             index = pfam_families.index(pfam_labels[i][0])
#             print('Sequence index', i,' and class number', class_num)
#             if index == class_num:
#                 print('Sequence found!')
#                 indices.append(i)
#                 class_seq.append(enc_sequences[i])
#                 class_labels.append(enc_labels[i])
#         
#         enc_sequences = np.delete(enc_sequences, indices, axis = 0)
#         enc_labels = np.delete(enc_labels, indices, axis = 0)
#         pfam_labels = np.delete(pfam_labels, indices, axis = 0)   
#         
#         if len(indices) != 0:
#             print('Saving new files!')
#             np.savez_compressed(pfam_batches_path + pfam_batches_list[file_index] + '/extracted-single-label/encoded-sequences.npz', enc_sequences)
#             np.savez_compressed(pfam_batches_path + pfam_batches_list[file_index] + '/extracted-single-label/encoded-labels.npz', enc_labels)
#             np.savez_compressed(pfam_batches_path + pfam_batches_list[file_index] + '/extracted-single-label/pfam-labels.npz', pfam_labels)
#         
#         if len(class_seq) == dict_counts[class_num]:
#             break
#         
#     print(len(class_seq), ' sequences extracted!')
#     
#     if not os.path.exists(classes_path + '{}/'.format(class_num)):
#         
#         os.mkdir(classes_path + '{}/'.format(class_num))
#         os.mkdir(classes_path + '{}/sequences/'.format(class_num))
#         os.mkdir(classes_path + '{}/labels/'.format(class_num))
#         
#         chunk_size = dict_thresh_counts[class_num]
#         
#         print(chunk_size)
#         b = 0
#         for j in range(0, len(class_seq), chunk_size):
#             np.savez_compressed(classes_path + '{}/sequences/batch-{}-sequences.npz'.format(class_num, b), class_seq[j:j + chunk_size])
#             np.savez_compressed(classes_path + '{}/labels/batch-{}-labels.npz'.format(class_num, b), class_labels[j:j + chunk_size])
#             b += 1
# 
# 
# =============================================================================

'''
4. create batches with equal distributions (one label per instance)
'''
# =============================================================================
# 
# classes_file_path = '../../Data/extracted-pfam-total-data/pfam-data-classes/'
# classes_files_list = sorted(os.listdir(classes_file_path))
# 
# stratified_dataset_path = '../../Data/stratified-dataset/single-label-batches/'
# 
# for batch_num in range(0, 6):
#     batch_seq = []
#     batch_labels = []
#     for class_num in range(996):
#         enc_sequences_list = sorted(os.listdir(classes_file_path + classes_files_list[class_num] + '/sequences/'))
#         enc_labels_list = sorted(os.listdir(classes_file_path + classes_files_list[class_num] + '/labels/'))
#         enc_seq = np.load(classes_file_path + classes_files_list[class_num] + '/sequences/' + enc_sequences_list[batch_num])
#         enc_label = np.load(classes_file_path + classes_files_list[class_num] + '/labels/' + enc_labels_list[batch_num])
#         
#         batch_seq.extend(enc_seq['arr_0'])
#         batch_labels.extend(enc_label['arr_0'])
#         print('Class-{} batch-{} sequences and labels loaded!'.format(class_num, batch_num + 1))
#         print(len(batch_seq))
#         print(len(batch_labels))
#         
#     print('Saving batch-{}'.format(batch_num + 1))
#     
#     if not os.path.exists(stratified_dataset_path + 'batch-{}/'.format(batch_num + 1)):
#          os.mkdir(stratified_dataset_path + 'batch-{}/'.format(batch_num + 1))
#          os.mkdir(stratified_dataset_path + 'batch-{}/sequences/'.format(batch_num + 1))
#          os.mkdir(stratified_dataset_path + 'batch-{}/labels/'.format(batch_num + 1))
#          
#     np.savez_compressed(stratified_dataset_path + 'batch-{}/sequences/enc-sequences.npz'.format(batch_num + 1), batch_seq)
#     np.savez_compressed(stratified_dataset_path + 'batch-{}/labels/enc-labels.npz'.format(batch_num + 1), batch_labels)
#     print('Batch-{} saved!'.format(batch_num + 1))
# 
# =============================================================================


'''
5. create batches with equal distributions (multiple labels per instance)
'''


# =============================================================================
# pfam_batches_path = '../../Data/extracted-pfam-total-data/extracted-pfam-data-batches/'
# pfam_batches_files_list = sorted(os.listdir(pfam_batches_path))
# 
# seq = []
# labels = []
# 
# for file_index in range(len(pfam_batches_files_list)):
#     seq_file = np.load(pfam_batches_path + pfam_batches_files_list[file_index] + '/extracted-multiple-labels/encoded-sequences.npz')
#     seq.extend(seq_file['arr_0'])
#     print(pfam_batches_files_list[file_index], 'appended!')
#     print('Sequences length', len(seq))
#     
#     encoded_label_file = np.load(pfam_batches_path + pfam_batches_files_list[file_index] + '/extracted-multiple-labels/encoded-labels.npz', allow_pickle= True)
#     labels.extend(encoded_label_file['arr_0'])
#     print(pfam_batches_files_list[file_index], 'appended!')
#     print('Labels length', len(labels))  
#     
# 
# stratified_dataset_path = '../../Data/stratified-dataset/multiple-label-batches/'
# 
# chunk_size = int(len(seq)/6)
# batch_num = 0
# 
# for i in range(0, len(labels), chunk_size):
#     if not os.path.exists(stratified_dataset_path + 'batch-{}/'.format(batch_num + 1)):
#         os.mkdir(stratified_dataset_path + 'batch-{}/'.format(batch_num + 1))
#         os.mkdir(stratified_dataset_path + 'batch-{}/sequences/'.format(batch_num + 1))
#         os.mkdir(stratified_dataset_path + 'batch-{}/labels/'.format(batch_num + 1))
#           
#     
#     print('Saving Batch-{}!'.format(batch_num + 1))
#     np.savez_compressed(stratified_dataset_path + 'batch-{}/sequences/enc-sequences.npz'.format(batch_num + 1), seq[i:i + chunk_size])
#     np.savez_compressed(stratified_dataset_path + 'batch-{}/labels/enc-labels.npz'.format(batch_num + 1), labels[i:i + chunk_size])
#     print('Batch-{} saved!'.format(batch_num + 1))
#     batch_num += 1
# 
# =============================================================================

