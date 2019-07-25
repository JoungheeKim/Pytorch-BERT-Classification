import os
import torch
from tqdm import tqdm
import torch.optim as optim
import logging
from torch import nn
import json
import numpy as np
from torch.optim import lr_scheduler

class Trainer():
    def __init__(self, model, crit, config, boost=False, device="cpu"):
        self.model = model
        self.crit = crit
        self.config = config
        self.boost = boost
        self.device = device
        self.gradient_accumulation_steps = config.gradient_accumulation_steps

        self.save_path = os.path.join(config.save_path)
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        device = config.device
        config.device = "gpu"
        with open(os.path.join(self.save_path, "config.json"), 'w') as outfile:
            json.dump(vars(config), outfile)
        config.device = device

        super().__init__()

        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.lower_is_better = True
        self.best = {'epoch': 0,
                     'config': config
                     }
        logging.info("##################### Init Trainer")

        self.history ={
            "train_loss" : [],
            "valid_loss" : [],
            "valid_correct": []
        }


    def get_best_model(self):
        self.model.load_state_dict(self.best['model'])
        return self.model

    def save_training(self, path):
        torch.save(self.best, path)

    def save_history(self, path):
        np.save(os.path.join(path, "train_loss"), self.history["train_loss"])
        np.save(os.path.join(path, "valid_loss"), self.history["valid_loss"])
        np.save(os.path.join(path, "valid_correct"), self.history["valid_correct"])

    def train(self, train, valid):
        '''
        Train with given train and valid iterator until n_epochs.
        If early_stop is set,
        early stopping will be executed if the requirement is satisfied.
        '''


        logging.info("run train")
        best_loss = float('Inf') * (1 if self.lower_is_better else -1)
        lowest_after = 0

        progress_bar = tqdm(range(self.best['epoch'], self.n_epochs),
                            desc='Training: ',
                            unit='epoch'
                            )

        optimizer = optim.Adam(
            [
                {"params": self.model.bert.parameters(), "lr": self.config.lr_main},
                {"params": self.model.fc.parameters(), "lr": self.config.lr}
            ])

        self.grobal_step = 0

        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        for idx in progress_bar:  # Iterate from 1 to n_epochs

            scheduler.step()

            avg_train_loss = self.train_epoch(train, optimizer)
            avg_valid_loss, avg_valid_correct = self.validate(valid)
            progress_bar.set_postfix_str('train_loss=%.4e valid_loss=%.4e valid_correct=%.4f min_valid_loss=%.4e' % (avg_train_loss,
                                                                                                                     avg_valid_loss,
                                                                                                                     avg_valid_correct,
                                                                                                                     best_loss))
            logging.debug('train_loss=%.4e valid_loss=%.4e valid_correct=%.4f min_valid_loss=%.4e' % (avg_train_loss,
                                                                                                      avg_valid_loss,
                                                                                                      avg_valid_correct,
                                                                                                      best_loss))
            self.history["train_loss"].append(avg_train_loss)
            self.history["valid_loss"].append(avg_valid_loss)
            self.history["valid_correct"].append(avg_valid_correct)
            self.save_history(self.save_path)

            if (self.lower_is_better and avg_valid_loss < best_loss) or \
               (not self.lower_is_better and avg_valid_loss > best_loss):
                # Update if there is an improvement.
                best_loss = avg_valid_loss
                lowest_after = 0

                self.best['model'] = self.model.state_dict()
                self.best['epoch'] = idx + 1

                # Set a filename for model of last epoch.
                # We need to put every information to filename, as much as possible.

                model_name = "model" + str(self.best['epoch']) + '.pwf'
                self.save_training(os.path.join(self.save_path, model_name))
            else:
                lowest_after += 1

                if lowest_after >= self.config.early_stop and \
                   self.config.early_stop > 0:
                    logging.debug("early stop")
                    progress_bar.close()
                    return self.history
        progress_bar.close()
        return self.history

    def train_epoch(self, train, optimizer):
        '''
        Train an epoch with given train iterator and optimizer.
        '''
        total_loss, total_count = 0, 0
        total_correct = 0
        avg_loss = 0
        avg_correct = 0

        progress_bar = tqdm(train,
                            desc='Training: ',
                            unit='batch'
                            )
        # Iterate whole train-set.
        optimizer.zero_grad()
        self.model.train()
        for step, batch in enumerate(progress_bar):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_id = batch

            label_hat = self.model(input_ids=input_ids,
                                   segment_ids=segment_ids,
                                   input_mask=input_mask,
                                   boost=self.boost)

            # Calcuate loss and gradients with back-propagation.
            loss = self.crit(label_hat, label_id)

            if self.gradient_accumulation_steps>1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()

            # Simple math to show stats.
            total_loss += float(loss)
            total_count += int(label_id.size(0))
            avg_loss = total_loss / total_count

            ps = torch.exp(label_hat)
            top_p, top_class = ps.topk(1, dim=1)

            equals = top_class == label_id.view(*top_class.shape)
            total_correct += torch.sum(equals).item()

            avg_correct = total_correct / total_count
            progress_bar.set_postfix_str('avg_loss=%.4e  correct=%.4f' % (avg_loss, avg_correct))

            # Take a step of gradient descent.
            if (step + 1) % self.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        progress_bar.close()

        return avg_loss

    def validate(self, valid):
        total_loss, total_count = 0, 0
        total_correct = 0
        avg_loss = 0
        avg_correct = 0

        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(valid, desc='Validation: ', unit='batch')
            # Iterate for whole valid-set.
            for batch in progress_bar:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_id = batch

                label_hat = self.model(input_ids=input_ids,
                                       segment_ids=segment_ids,
                                       input_mask=input_mask,
                                       boost=self.boost)
                loss = self.crit(label_hat, label_id)

                total_loss += float(loss)
                total_count += int(label_id.size(0))
                avg_loss = total_loss / total_count

                ps = torch.exp(label_hat)
                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == label_id.view(*top_class.shape)
                total_correct += torch.sum(equals).item()

                avg_correct = total_correct / total_count
                progress_bar.set_postfix_str('avg_loss=%.4e  correct=%.4f' % (avg_loss, avg_correct))

            progress_bar.close()
        self.model.train()
        return avg_loss, avg_correct



