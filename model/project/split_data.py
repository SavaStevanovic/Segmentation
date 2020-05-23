import os
from shutil import copyfile
from sklearn.model_selection import train_test_split


def copy_files(dataset, input_folder, output_folder, folder):
    os.makedirs(os.path.join(output_folder, folder), exist_ok=True)
    for f in dataset:
        file_path = os.path.join(input_folder, f)
        out_file = os.path.join(output_folder, folder, f)
        copyfile(file_path, out_file)

output_folder = '/Data/dataset_10224_raw/raw_split/'
input_folder = '/Data/dataset_10224_raw/raw/'

files = os.listdir(input_folder)
train, val_test = train_test_split(files, test_size=0.2, random_state=1)
validation, test = train_test_split(val_test, test_size=0.5, random_state=2)

copy_files(train, input_folder, output_folder, 'train')
copy_files(validation, input_folder, output_folder, 'validation')
copy_files(test, input_folder, output_folder, 'test')
