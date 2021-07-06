import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import os

# open file with unique pfam labels
with open("../../Data/pfam-unique-families.json") as read_file:
    pfam_unique_labels = json.load(read_file)
    
pfam_unique_labels = np.asarray(pfam_unique_labels)

# convert each pfam label to integer format

label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(pfam_unique_labels)

report_file =  open('../../Data/pfam-integer-mapping.txt', "w")
report_file.write("Pfam\t\tInteger\n")

for index in range(len(pfam_unique_labels)):
    line = [pfam_unique_labels[index], '\t\t', str(integer_labels[index]),"\n"]
    line = ''.join(line)
    report_file.write(line)
    report_file.writable()

report_file.close()


# load the extracted label dataset
pfam_labels = np.load('../../Data/extracted-pfam-total-data/pfam-labels-crop.npz', allow_pickle=True)
pfam_labels = pfam_labels['arr_0']

# convert the selected labels to integer format and further convert to tuple 
# to convert it into one-hot format
for i in range(len(pfam_labels)):
    for j in range(len(pfam_labels[i])):
        index = np.where(pfam_labels[i][j] == pfam_unique_labels)[0][0]
        pfam_labels[i][j] = integer_labels[index]
    pfam_labels[i] = tuple(pfam_labels[i])

del pfam_unique_labels, integer_labels

# convert to one-hot encoding format     
mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(pfam_labels)

print(encoded_labels.shape)
#total_pfam_encoded_labels = total_pfam_encoded_labels.astype('float32')

# save encoded labels
np.savez_compressed('../../Data/extracted-pfam-total-data/encoded-labels.npz', encoded_labels)

batches_path = '../../Data/extracted-pfam-total-data/extracted-pfam-data-batches/'
batch_num = 1
chunk_size = 1000000

for batch in range(0, len(encoded_labels), chunk_size):
    np.savez_compressed(batches_path + 'batch-{}/encoded-labels.npz'.format(batch_num), encoded_labels[batch:batch + chunk_size])

    batch_num += 1