import json
import torch
import shutil
from sklearn import metrics
from timeit import default_timer as timer

# Train function
def train_step(args, language_model, vision_model, classifier, loss_function, optimizer, dataloader, print_batch=50):
    '''
    Perform training process including forward and backward pass for one epoch.

        Parameters:
            args (Namespace): hyperparameters  
            language_model (nn.Module): pre-trained language model
            classifier (nn.Module): classifier layers
            loss_function: loss function
            optimizer: optimizer algorithm
            dataloader: pytorch dataloader
            print_batch (int): print out train time and loss every number of print_batch
        Returns:
            time_total (float): total time to train one epoch
            loss_total (float): total train loss for one epoch
            loss_average (float): average train loss for one epoch
    '''
    language_model.train()
    vision_model.train()
    classifier.train()
    loss_total, time_total = 0, 0
    for batch_index, data in enumerate(dataloader):
        time_start = timer()
        images, captions = data['images'].to(args.DEVICE), data['captions'].to(args.DEVICE)
        masks, labels = data['masks'].to(args.DEVICE), data['labels'].to(args.DEVICE)
        language_model_output = language_model(input_ids = captions, attention_mask=masks).last_hidden_state[:, 0, :]
        vision_model_output = vision_model(images)
        logit = classifier(language_model_output, vision_model_output)
        loss = loss_function(logit, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss
        time_end = timer()
        time_total += (time_end - time_start)
        if batch_index % 50 == 0:
            print(f'Time to train {batch_index} batches: {time_total:.5f} secs | Loss at batch {batch_index}: {loss:.5f}')
    loss_average = loss_total / len(dataloader)
    return time_total, loss_total, loss_average

# Test function
def test_step(args, language_model, vision_model, classifier, dataloader, print_batch=30):
    '''
    Perform testing process including forward pass.

        Parameters:
            args (Namespace): hyperparameters  
            language_model (nn.Module): pre-trained language model
            vision_model (nn.Module): pre-trained vision model
            classifier (nn.Module): classifier layers
            dataloader: pytorch dataloader
            print_batch (int): print out notification every number of print_batch
        Returns:
            labels_true (list): list of true labels (int)
            labels_pred (list): list of predicted labels (int)
    '''
    labels_true, labels_pred = [], []
    language_model.eval()
    vision_model.eval()
    classifier.eval()
    with torch.inference_mode():
        for batch_index, data in enumerate(dataloader):
            images, captions = data['images'].to(args.DEVICE), data['captions'].to(args.DEVICE)
            masks, labels = data['masks'].to(args.DEVICE), data['labels'].to(args.DEVICE)

            language_model_output = language_model(input_ids = captions, attention_mask=masks).last_hidden_state[:, 0, :]
            vision_model_output = vision_model(images)

            logit = classifier(language_model_output, vision_model_output)
            batch_labels_pred = logit.argmax(dim=1)

            labels_true.extend(labels.tolist())
            labels_pred.extend(batch_labels_pred.tolist())
            if batch_index % 30 == 0:
                print(f'Finished testing {batch_index} batches')
    return labels_true, labels_pred

# get evaluation metrics function
def get_metrics(labels_true, labels_pred, labels_to_ids):
    '''
    Get evaluation metrics from true and predicted labels.

        Parameters:
            labels_true (list): list of true labels (int)
            labels_pred (list): list of predicted labels (int)
            labels_to_ids (dict): mapping of labels to ids
        Returns:
            micro_precision (float): Micro precision
            micro_recall (float): Micro recall
            micro_f1 (float): Micro F1-score
            cls_report (float): Sklearn classification report
    '''
    micro_precision = metrics.precision_score(labels_true, labels_pred, average='micro', zero_division=0.0)
    micro_recall = metrics.recall_score(labels_true, labels_pred, average='micro', zero_division=0.0)
    micro_f1 = metrics.f1_score(labels_true, labels_pred, average='micro', zero_division=0.0)
    cls_report = metrics.classification_report(labels_true, labels_pred, target_names=labels_to_ids, zero_division=0.0, digits=5)
    return micro_precision, micro_recall, micro_f1, cls_report

# Export result function
def export_result(phase, labels_pred, ids_to_labels):
    '''
    Export result into a json file and then zip it.

        Parameters:
            phase (str): specify the competition phase
            labels_pred (list): list of predicted labels (int)
            ids_to_labels (dict): mapping of ids to labels
        Returns:
            None
    '''
    output = {"results": {}, "phase": phase}
    output['results'] = {str(index):ids_to_labels[value] for index, value in enumerate(labels_pred)}
    content = json.dumps(output, indent=4)
    with open("results.json", "w") as outfile: outfile.write(content)
    shutil.make_archive('results', 'zip', '', 'results.json')