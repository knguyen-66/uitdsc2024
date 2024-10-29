from argparse import Namespace
import sys
from typing import Literal
import timm
import tqdm
import torch
import random
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from transformers import AutoTokenizer, AutoModel, ResNetModel

from make_dummy_data import make_dummy_data
from data import load_data, get_labels, RawData, data_process
from utilities import train_step, test_step, get_metrics, export_result


class SingleClassificationLayers(nn.Module):
    def __init__(self, args, labels_to_ids):
        super().__init__()

        if args.SINGLE_MODEL_TYPE == "PLM":
            self.cls_layers = nn.Sequential(
                # nn.Dropout(args.CLS_DROPOUT),
                # nn.Linear(args.PLM_OUTPUT_SIZE, 100),
                # nn.Linear(100, len(labels_to_ids)),
                nn.Linear(args.PLM_OUTPUT_SIZE, len(labels_to_ids)),
            )
        elif args.SINGLE_MODEL_TYPE == "PVM":
            self.cls_layers = nn.Sequential(
                # nn.Dropout(args.CLS_DROPOUT),
                # nn.Linear(args.PVM_OUTPUT_SIZE, 100),
                # nn.Linear(args.PVM_OUTPUT_SIZE, len(labels_to_ids)),
                nn.Linear(args.PVM_OUTPUT_SIZE, len(labels_to_ids)),
            )

    def forward(self, input):
        logit = self.cls_layers(input)
        return logit


class ClassificationLayers(nn.Module):
    def __init__(self, args, labels_to_ids):
        super().__init__()
        # self.linear_plm = nn.Linear(args.PLM_OUTPUT_SIZE, 100)
        # self.linear_pvm = nn.Linear(args.PVM_OUTPUT_SIZE, 100)
        # self.linear_final = nn.Linear(200, len(labels_to_ids))
        self.cls_layers = nn.Sequential(
            # nn.Linear(args.PLM_OUTPUT_SIZE + args.PVM_OUTPUT_SIZE, args.PLM_OUTPUT_SIZE + args.CLS_SIZE),
            # # nn.Tanh(),
            # nn.Dropout(args.CLS_DROPOUT),
            # nn.Linear(args.PLM_OUTPUT_SIZE + args.CLS_SIZE, args.CLS_SIZE),
            # # nn.Tanh(),
            # nn.Dropout(args.CLS_DROPOUT),
            # nn.Linear(args.CLS_SIZE, len(labels_to_ids))
            nn.Linear(args.PLM_OUTPUT_SIZE + args.PVM_OUTPUT_SIZE, len(labels_to_ids))
        )

    def forward(self, plm_input, pvm_input):
        # plm_input = self.linear_plm(plm_input)
        # pvm_input = self.linear_pvm(pvm_input)
        input = torch.cat((plm_input, pvm_input), dim=-1)
        logit = self.cls_layers(input)
        return logit


def get_example_args():
    return {
        "SINGLE_MODEL": [bool, [True, False]],
        "SINGLE_MODEL_TYPE": [str, ["PLM", "PVM"]],
        "CLS_DROPOUT": [float, []],
        "EPOCHS": [int, []],
        "PLM": [str, []],
        "PLM_OUTPUT_SIZE": [int, []],
        "PLM_MAX_TOKEN": [int, []],
        "PVM": [str, []],
        "PVM_OUTPUT_SIZE": [int, []],
        "IMAGE_SIZE": [int, []],
        "PLM_LEARNING_RATE": [float, []],
        "PVM_LEARNING_RATE": [float, []],
        "CLS_LEARNING_RATE": [float, []],
        "CLS_SIZE": [int, []],
        "TRAIN_MODE": [str, ["train", "test"]],
        "TRAIN_BATCH_SIZE": [int, []],
        "TEST_BATCH_SIZE": [int, []],
        "RANDOM_SEED": [int, [2022, 2023, 2024]],
    }


def execute(data_path: str, args: dict):
    print("\n------ Preparing data ------")

    args["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    args = Namespace(**args)

    random.seed(args.RANDOM_SEED)
    np.random.seed(args.RANDOM_SEED)
    torch.manual_seed(args.RANDOM_SEED)
    torch.cuda.manual_seed_all(args.RANDOM_SEED)

    data_subpath = "data"
    if args.TRAIN_MODE == "train":
        make_dummy_data(data_path, 1000)
        data_subpath = "dummy_data"

    images_train_dir, images_train, annotations_train = load_data(data_path, data_subpath, "train")
    images_dev_dir, images_dev, annotations_dev = load_data(data_path, data_subpath, "dev")

    print(f"Current directory: {data_path}")
    print(f"No of train samples: {len(images_train)}")
    print(f"No of dev samples: {len(images_dev)}")

    labels, labels_to_ids, ids_to_labels = get_labels(annotations_train)

    print(f"All labels: {labels}")
    print(f"labels to ids: {labels_to_ids}")
    print(f"ids to labels: {ids_to_labels}")

    trainset_raw = RawData(images_train_dir, annotations_train, labels_to_ids)
    devset_raw = RawData(images_dev_dir, annotations_dev, labels_to_ids)

    tokenizer = AutoTokenizer.from_pretrained(args.PLM)
    trainset_dataloader = DataLoader(
        trainset_raw,
        batch_size=args.TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: data_process(batch, tokenizer, args.PLM_MAX_TOKEN, args.IMAGE_SIZE),
    )
    devset_dataloader = DataLoader(
        devset_raw,
        batch_size=args.TEST_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: data_process(batch, tokenizer, args.PLM_MAX_TOKEN, args.IMAGE_SIZE),
    )

    print(f"Number of samples in train set: {len(trainset_raw)}")
    print(f"Number of samples in dev set: {len(devset_raw)}\n")
    print(f"Number of train batches: {len(trainset_dataloader)}")
    print(f"Number of dev batches: {len(devset_dataloader)}")

    print("\n------ Preparing model ------")

    training_param = []

    if args.SINGLE_MODEL:
        classifier = SingleClassificationLayers(args, labels_to_ids).to(args.DEVICE)
        if args.SINGLE_MODEL_TYPE == "PLM":
            vision_model = None
            language_model = AutoModel.from_pretrained(args.PLM, return_dict=True).to(args.DEVICE)
            language_model_param = tuple(language_model.named_parameters())
            training_param += [{"params": [param for name, param in language_model_param], "lr": args.PLM_LEARNING_RATE}]
        elif args.SINGLE_MODEL_TYPE == "PVM":
            language_model = None
            vision_model = timm.create_model(args.PVM, pretrained=True, num_classes=0).to(args.DEVICE)
            vision_model_param = tuple(vision_model.named_parameters())
            training_param += [{"params": [param for name, param in vision_model_param], "lr": args.PVM_LEARNING_RATE}]

    else:
        language_model = AutoModel.from_pretrained(args.PLM, return_dict=True).to(args.DEVICE)
        vision_model = timm.create_model(args.PVM, pretrained=True, num_classes=0).to(args.DEVICE)
        classifier = ClassificationLayers(args, labels_to_ids).to(args.DEVICE)

        language_model_param = tuple(language_model.named_parameters())
        vision_model_param = tuple(vision_model.named_parameters())

        training_param += [
            {"params": [param for name, param in language_model_param], "lr": args.PLM_LEARNING_RATE},
            {"params": [param for name, param in vision_model_param], "lr": args.PVM_LEARNING_RATE},
        ]

    classifier_param = tuple(classifier.named_parameters())
    training_param += [{"params": [param for name, param in classifier_param], "lr": args.CLS_LEARNING_RATE}]

    optimizer = optim.Adam(training_param)
    loss_function = nn.CrossEntropyLoss()

    print("\n------ Starting training process ------")

    best_f1 = 0
    for epoch in tqdm.trange(args.EPOCHS, file=sys.stdout):
        print(f"\n\nEpoch {epoch}:")
        print("-----------")
        time_total, loss_total, loss_average = train_step(
            args, language_model, vision_model, classifier, loss_function, optimizer, trainset_dataloader, 50
        )
        print("-----------")
        print(f"Total train time: {time_total:.5f} secs")
        print(f"Total loss: {loss_total:.5f}")
        print(f"Average loss: {loss_average:.5f}")

        print("\nEvaluating.......")
        labels_true, labels_pred = test_step(args, language_model, vision_model, classifier, devset_dataloader, 30)
        print("****Finished testing****")

        if args.TRAIN_MODE == "train":
            micro_precision, micro_recall, micro_f1, cls_report = get_metrics(labels_true, labels_pred, labels_to_ids)
            print("\n[+] METRICS:")
            print(f"Micro precision: {micro_precision:.5f}")
            print(f"Micro recall: {micro_recall:.5f}")
            print(f"Micro F1: {micro_f1:.5f}")
            print(f"Classification report:\n{cls_report}")

            if micro_f1 > best_f1:
                best_f1 = micro_f1
                export_result("dev", labels_pred, ids_to_labels)
        else:
            export_result("dev", labels_pred, ids_to_labels, args, epoch)


if __name__ == "__main__":
    args = {
        "SINGLE_MODEL": True,
        "SINGLE_MODEL_TYPE": "PVM",  # PLM / PVM
        "CLS_DROPOUT": 0.15,
        "EPOCHS": 5,
        # try mutiple PLMS
        "PLM": "uitnlp/visobert",
        "PLM_OUTPUT_SIZE": 768,
        "PLM_MAX_TOKEN": 150,
        # try mutiple PVMS
        "PVM": "timm/resnet152.a1h_in1k",
        "PVM_OUTPUT_SIZE": 2048,
        "IMAGE_SIZE": 224,
        # lr
        "PLM_LEARNING_RATE": 1e-5,
        "PVM_LEARNING_RATE": 1e-5,
        "CLS_LEARNING_RATE": 1e-4,
        "CLS_SIZE": 512,
        # batch size
        "TRAIN_MODE": "train",  # train / test
        "TRAIN_BATCH_SIZE": 32,
        "TEST_BATCH_SIZE": 16,
        "RANDOM_SEED": 2024,  # 2023, 2022
    }
    execute(".", args)
