import os
import json
import shutil
import pathlib

def load_data(path, data_folder_name:str, dataset:str):
    images_dir, annotations_dir = os.path.join(path, data_folder_name, 'images'), os.path.join(path, data_folder_name, 'annotations')
    images_dataset_dir = os.path.join(images_dir, dataset)
    annotations_dir = os.path.join(annotations_dir, f'{dataset}.json')
    with open(annotations_dir, 'r') as train:
        annotations = json.load(train)
    output = (images_dataset_dir, annotations)
    return output

def make_dummy_data_dir(path):
    dummy_data_path = os.path.join(path, 'dummy_data')
    dummy_images_dir = os.path.join(dummy_data_path, 'images')
    dummy_annotations_dir = os.path.join(dummy_data_path, 'annotations')

    dummy_images_train_dir = os.path.join(dummy_images_dir, 'train')
    dummy_images_dev_dir = os.path.join(dummy_images_dir, 'dev')

    pathlib.Path(dummy_data_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(dummy_images_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(dummy_annotations_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(dummy_images_train_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(dummy_images_dev_dir).mkdir(parents=True, exist_ok=True)

    return dummy_annotations_dir, dummy_images_train_dir, dummy_images_dev_dir

def get_dummy_labels(annotations, no_of_samples):
    original_labels, dummy_labels = {'not-sarcasm': 0, 'text-sarcasm': 0, 'multi-sarcasm': 0, 'image-sarcasm': 0}, {}
    for annotations_index in annotations: original_labels[annotations[annotations_index]['label']] += 1
    label_keys, label_values = list(original_labels.keys()), list(original_labels.values())
    for key, value in zip(label_keys, label_values):
        label_num = round((value / sum(label_values)) * no_of_samples)
        dummy_labels[key] = label_num
    return dummy_labels

def make_dummy_data(path, no_of_samples):
    images_train_dir, annotations_train = load_data(path, 'data', 'train')
    dummy_annotations_dir, dummy_images_train_dir, dummy_images_dev_dir = make_dummy_data_dir(path)
    dummy_labels = get_dummy_labels(annotations_train, no_of_samples)

    dummy_annotations_train_json_path = os.path.join(dummy_annotations_dir, 'train.json')
    dummy_annotations_dev_json_path = os.path.join(dummy_annotations_dir, 'dev.json')

    check_dict = {key:0 for key in dummy_labels.keys()}
    dummy_annotations_train, dummy_annotations_dev = {}, {}
    dummy_annotations_train_index, dummy_annotations_dev_index = 0, 0


    for annotations_index in annotations_train:
        annotations = annotations_train[annotations_index]
        image, label = annotations['image'], annotations['label']
        current_image = os.path.join(images_train_dir, image)
        dummy_image_train = os.path.join(dummy_images_train_dir, image)
        dummy_image_dev = os.path.join(dummy_images_dev_dir, image)
        
        if check_dict[label] == dummy_labels[label] and os.path.exists(dummy_image_train) == False:
            shutil.copy(current_image, dummy_images_train_dir)
            dummy_annotations_train[str(dummy_annotations_train_index)] = annotations
            dummy_annotations_train_index += 1
        elif check_dict[label] < dummy_labels[label] and os.path.exists(dummy_image_dev) == False:
            shutil.copy(current_image, dummy_images_dev_dir)
            check_dict[label] += 1
            dummy_annotations_dev[str(dummy_annotations_dev_index)] = annotations
            dummy_annotations_dev_index += 1

    if dummy_annotations_train: 
        dummy_annotations_train_json = json.dumps(dummy_annotations_train, indent=4)
        with open(dummy_annotations_train_json_path, "w") as train: train.write(dummy_annotations_train_json)
    if dummy_annotations_dev: 
        dummy_annotations_dev_json = json.dumps(dummy_annotations_dev, indent=4)
        with open(dummy_annotations_dev_json_path, "w") as dev: dev.write(dummy_annotations_dev_json)

    assert check_dict == dummy_labels
    