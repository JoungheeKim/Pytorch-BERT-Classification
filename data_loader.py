import csv
import torch
import logging
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class MyDataLoader():
    def __init__(self,
                 train_path,
                 valid_path,
                 tokenizer,
                 max_length=512):

        self.tokenizer = tokenizer
        self.train_path = train_path
        self.valid_path = valid_path
        self.max_length = max_length

    def get_train_valid_data(self):
        train, train_label_list = get_data(self.train_path)
        valid, valid_label_list = get_data(self.valid_path)

        label_list = list(set(train_label_list + valid_label_list))

        train = get_dataset(train, label_list, self.tokenizer, self.max_length)
        valid = get_dataset(valid, label_list, self.tokenizer, self.max_length)

        return train, valid, len(label_list)

    def convert_ids_to_vector(self, data, model, batch_size=1, device="cpu"):
        model.eval()

        dataLoader = DataLoader(data, batch_size=batch_size, shuffle=False)
        progress_bar = tqdm(dataLoader, desc='boosting: ', unit='dataset')
        pooled_vector = []
        with torch.no_grad():
            for idx, batch in enumerate(progress_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_id = batch
                pooled_output = model.ids_to_vector(input_ids=input_ids,
                                                    segment_ids=segment_ids,
                                                    input_mask=input_mask)
                pooled_vector.append(pooled_output.to("cpu"))
        pooled_vector = torch.cat(pooled_vector)
        dummy_data = torch.zeros(len(data))
        progress_bar.close()

        model.train()
        return TensorDataset(pooled_vector, dummy_data, dummy_data, data.tensors[3])


def get_data(data_path):
    data, label_list = [], []
    with open(data_path, encoding="UTF8") as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(reader):
            label = line[1]
            data.append((line[0], label))
            if not label in label_list:
                label_list.append(label)
    return data, label_list

def get_dataset(data, label_list, tokenizer, max_length=512):
    features = convert_examples_to_features(data_list=data,
                                            label_list=label_list,
                                            tokenizer=tokenizer,
                                            max_length=max_length)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


def convert_examples_to_features(data_list, label_list, tokenizer, max_length=512):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for idx, (data, label) in enumerate(data_list):
        if idx % 10000 == 0:
            logging.info("Writing example %d" % (idx))

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        tokens = tokenizer.tokenize(data)
        _truncate_seq(tokens, max_length-2)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        label_id = label_map[label]

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id)
        )
    return features

def _truncate_seq(tokens_a, max_length):
    while True:
        total_length = len(tokens_a)
        if total_length <= max_length:
            break
        tokens_a.pop()

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id




