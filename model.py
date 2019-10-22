import torch
import torch.nn as nn
import torch.nn.functional as F

class MyClassifier(nn.Module):
    def __init__(self, bert, num_class, bert_finetuning=False ,dropout_p=0.03, device="cpu"):
        super(MyClassifier, self).__init__()

        self.device = device
        self.bert = bert
        self.dropout_p = dropout_p
        self.num_class = num_class
        self.bert_finetuning = bert_finetuning

        # BERT hidden size -- D : 768
        bert_hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(bert_hidden_size, self.num_class)
        self.softmax = nn.LogSoftmax(dim=-1)

    def ids_to_vector(self, input_ids, segment_ids=None, input_mask=None, boost=False):

        if boost and not self.bert_finetuning:
            return input_ids

        # input_ids : (B, D)
        # B : batch size,
        # D : dimenstion of tokens - 512

        """
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).
            [batch_size, hidden_size]
        """
        if self.training and self.bert_finetuning:
            self.bert.train()
            _, pooled_output = self.bert(input_ids=input_ids,
                                         token_type_ids=segment_ids,
                                         attention_mask=input_mask)
        else:
            self.bert.eval()
            with torch.no_grad():
                _, pooled_output = self.bert(input_ids=input_ids,
                                             token_type_ids=segment_ids,
                                             attention_mask=input_mask)
        return pooled_output

    def forward(self, input_ids, segment_ids=None, input_mask=None, boost=False):
        # input_ids : (B, D)
        # B : batch size,
        # D : dimenstion of tokens - 512

        x = self.ids_to_vector(input_ids, segment_ids, input_mask, boost)
        # x : (B, D)
        # B : batch size,
        # D : dimenstion of bert hidden - 768

        x = self.dropout(x)
        x = self.fc(x)
        logit = self.softmax(x)

        return logit



