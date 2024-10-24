import os
import cv2
import json
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# load data function
def load_data(current_path, data_foler:str, dataset:str):
    '''
    Load specific dataset from given path.

        Parameters:
            current_path (str or PathObj): path to the directory containing the data folder
            data_foler (str): name of data folder
            dataset (str): specific dataset e.g. 'train', 'test', or 'dev'
        Returns:
            images_dataset_dir (str): path to the directory containing image folder
            images (str): path to the directory containing the images
            annotations (obj): annotations for the image containing: {image, caption, label}
    '''
    images_dir, annotations_dir = os.path.join(current_path, data_foler, 'images'), os.path.join(current_path, data_foler, 'annotations')
    images_dataset_dir = os.path.join(images_dir, dataset)
    annotations_dir = os.path.join(annotations_dir, f'{dataset}.json')
    images = os.listdir(images_dataset_dir)
    with open(annotations_dir, 'r') as file: annotations = json.load(file)
    return images_dataset_dir, images, annotations

# get labels function
def get_labels(annotations):
    '''
    Extract labels from the given annotations of a specific dataset.

        Parameters:
            annotations (obj): annotations of a specific dataset
        Returns:
            labels (set): all the labels of a specific dataset
            labels_to_ids (dict): mapping of labels to ids
            ids_to_labels (dict): mapping of ids to labels
    '''
    labels = set()
    for annotations_index in annotations: labels.add(annotations[annotations_index]['label'])
    labels_to_ids = {label:index for index, label in enumerate(labels)}
    ids_to_labels = {index:label for label, index in labels_to_ids.items()}
    return labels, labels_to_ids, ids_to_labels

# Class to create raw dataset
class RawData(Dataset):
    def __init__(self, images_path, annotations, labels_to_ids):
        self.images_path = images_path
        self.annotations = annotations
        self.labels_to_ids = labels_to_ids
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.annotations[str(idx)]['image'])
        image = np.array(plt.imread(image_path), copy=True)
        caption = self.annotations[str(idx)]['caption']
        if self.annotations[str(idx)]['label'] is not None: label = self.labels_to_ids[self.annotations[str(idx)]['label']]
        else: label = 0
        data = {
            'image': image, # a 3 dimensional numpy array (height, width, chanel)
            'caption': caption, # caption of the image
            'label': label # a label for sarcasm
        }
        return data



# data preprocessing function
def data_process(batch, tokenizer, max_token, image_size):
    '''
    Data preprocessing function as a value for collate_fn param of pytorch DataLoader class

        Parameters:
            batch (obj): current batch
            tokenizer: huggingface tokenizer
            transform: torchvison transform module
            max_token (int): maximum number of embbeding tokens for tokenizer
            image_size (int): size to rescale image (for both height and width)
        Returns:
            data_final (obj): return a dictionary containing tensors of the following features: images, captions, masks, labels
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)) # size is a hyperparam
    ])

    images, captions, masks, labels = [], [], [], []
    for data in batch:
        image, caption, label = data['image'], data['caption'], data['label']
        if len(image.shape) < 3: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = transform(image).tolist()
        tokenizer_ouput = tokenizer(caption, padding='max_length', truncation=True,max_length=max_token)
        caption, mask = tokenizer_ouput.input_ids, tokenizer_ouput.attention_mask

        images.append(image)
        captions.append(caption)
        masks.append(mask)
        labels.append(label)
    data_final = {'images': torch.tensor(images), 'captions': torch.tensor(captions),
                  'masks': torch.tensor(masks), 'labels': torch.tensor(labels)}
    return data_final