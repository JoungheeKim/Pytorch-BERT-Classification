import pandas as pd
import numpy
from argparse import ArgumentParser
from data_loader import MyDataLoader
from torch import nn
import torch
import logging
import os
import tools
#from pytorch_pretrained_bert import BertTokenizer, BertModel
from model import MyClassifier
from trainer import Trainer
from torch.utils.data import DataLoader
from pytorch_transformers import BertTokenizer, BertModel


def build_parser():
    parser = ArgumentParser()

    ##Common option
    parser.add_argument("--device", dest="device", default="gpu")

    ##Loader option
    parser.add_argument("--train_path", dest="train_path", default="source/train.csv")
    parser.add_argument("--valid_path", dest="valid_path", default="source/test.csv")
    parser.add_argument("--max_length", dest="max_length", default=512, type=int)
    parser.add_argument("--save_path", dest="save_path", default="model")

    ##Model option
    parser.add_argument("--bert_name", dest="bert_name", default="bert-base-uncased")
    parser.add_argument("--bert_finetuning", dest="bert_finetuning", default=False, type=bool)
    parser.add_argument("--dropout_p", dest="dropout_p", default=0.1, type=int)

    ##Train option
    parser.add_argument("--boost", dest="boost", default=True, type=bool)
    parser.add_argument("--n_epochs", dest="n_epochs", default=10, type=int)
    parser.add_argument("--lr_main", dest="lr_main", default=0.00001, type=int)
    parser.add_argument("--lr", dest="lr", default=0.001, type=int)
    parser.add_argument("--early_stop", dest="early_stop", default=2, type=int)
    parser.add_argument("--batch_size", dest="batch_size", default=16, type=int)
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--num_class", dest="num_class", type=int)

    config = parser.parse_args()
    return config

def run(config):
    def _print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))

    _print_config(config)

    if not logging.getLogger() == None:
        for handler in logging.getLogger().handlers[:]:  # make a copy of the list
            logging.getLogger().removeHandler(handler)

    if not os.path.isdir(config.save_path):
        os.mkdir(config.save_path)
    all_subdir = [int(s) for s in os.listdir(config.save_path) if os.path.isdir(os.path.join(config.save_path, str(s)))]
    max_dir_num = 0
    if all_subdir:
        max_dir_num = max(all_subdir)
    max_dir_num += 1
    config.save_path = os.path.join(config.save_path, str(max_dir_num))
    os.mkdir(config.save_path)

    logging.basicConfig(filename=os.path.join(config.save_path, 'train_log'),
                        level=tools.LOGFILE_LEVEL,
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(tools.CONSOLE_LEVEL)
    logging.getLogger().addHandler(console)

    logging.info("##################### Start Training")
    logging.debug(vars(config))

    logging.info("##################### Start Load BERT MODEL")
    tokenizer = BertTokenizer.from_pretrained(config.bert_name)
    bert = BertModel.from_pretrained(config.bert_name)
    bert.to(config.device)

    ##load data loader
    logging.info("##################### Load DataLoader")
    loader = MyDataLoader(train_path=config.train_path,
                          valid_path=config.valid_path,
                          max_length=config.max_length,
                          tokenizer=tokenizer)

    train, valid, num_class = loader.get_train_valid_data()
    logging.info("##################### Train Dataset size : [" + str(len(train)) + "]")
    logging.info("##################### Valid Dataset size : [" + str(len(valid)) + "]")
    logging.info("##################### class size : [" + str(num_class) + "]")

    #modified batch size
    config.batch_size = config.batch_size // config.gradient_accumulation_steps
    logging.info("##################### Modified batch size : [" + str(config.batch_size) + "]")

    logging.info("##################### Load 'HAN' Model")
    model = MyClassifier(bert=bert,
                         num_class=num_class,
                         bert_finetuning=config.bert_finetuning,
                         dropout_p=config.dropout_p,
                         device=config.device
                         )
    model.to(config.device)
    crit = nn.NLLLoss()
    trainer = Trainer(model=model,
                      crit=crit,
                      config=config,
                      boost=config.boost,
                      device=config.device)

    # If bert fine-tuning process is not necessary, convert text into vectors by using bert to make whole process fast
    if config.boost and not config.bert_finetuning:
        logging.info("##################### Transform Dataset into Vectors by using BERT")
        train = loader.convert_ids_to_vector(data=train,
                                             model=model,
                                             batch_size=config.batch_size,
                                             device=config.device)
        valid = loader.convert_ids_to_vector(data=valid,
                                             model=model,
                                             batch_size=config.batch_size,
                                             device=config.device)

    train = DataLoader(dataset=train, batch_size=config.batch_size, shuffle=True)
    valid = DataLoader(dataset=valid, batch_size=config.batch_size, shuffle=True)

    history = trainer.train(train, valid)
    return history

if __name__ == "__main__":
    ##load config files
    config = build_parser()
    config.device = torch.device(
        "cuda" if torch.cuda.is_available() and (config.device == 'gpu' or config.device == 'cuda') else "cpu")
    run(config)