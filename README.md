# Pytorch-BERT-Classification
This is pytorch simple implementation of [Pre-training of Deep Bidirectional Transformers for
Language Understanding (BERT)](https://arxiv.org/pdf/1810.04805.pdf) by using awesome [pytorch BERT library](https://github.com/huggingface/pytorch-transformers)

### Dataset
1. IMDB(Internet Movie Database)
To test model, I use a dataset of 50,000 movie reviews taken from IMDb. 
It is divied into 'train', 'test' dataset and each data has 25,000 movie reviews and labels(positive, negetive).
You can access to dataset with this [link](http://ai.stanford.edu/~amaas/data/sentiment/)

2. Naver Movie review
[link](https://github.com/e9t/nsmc/)

3. KorQuAD(The Korean Question Answering Dataset)
[link](https://korquad.github.io/)

### How to use it?
Follow the example

#### 1 Train Model
There is a lot of options to check.
1. train_path : A File to train model
2. valid_path : A File to valid model
3. max_length :  Maximum length of word to analysis (BERT model restrict this parameter under 512) 
4. save_path : A Path to save result of BERT classfier model
5. bert_name : The name of pretrained BERT model. Defalut : bert-base-uncased ( More information about pytorch-BERT model can be found in this [link](https://github.com/google-research/bert) )
6. bert_finetuning : If you want to fintune BERT model with classfier layer, set "True" for this option
7. dropout_p : Drop probability of BERT result vector before enter to classfier layer
8. boost : If you don't need to fine tune BERT, you can make model faster to preconvert tokens to BERT result vectors. 
9. n_epochs : A number of epoches to train
10. lr : learning rate of classfier layer
10. lr_main : learning rate of BERT for fine tune
11. early_stop : A early_stop condition. If you don't want to use this options, put -1
12. batch_size : Batch size to train
13. gradient_accumulation_steps : BERT is very heavy model to handle large batch size with light GPU. So I implement gradient accumulation to handle samller batch size but almost same impact of using large batch size

```python
python train.py --train_path source/train.csv --valid_path source/test.csv --batch_size 16 --gradient_accumulation_steps 4 --boost True 
```

### Result
Result with hyper parameter settings

| BERT finetune | Max token Length | Best Epoch | train loss | valid loss | valid accuracy |
|---------------|:----------------:|:----------:|:----------:|:----------:|:--------------:|
| True          |        256       |     1      |   0.0169   |   0.0129   |     0.9181     |
| True          |        512       |     1      |   0.0151   |   0.0112   |     0.9292     |
| False         |        256       |     10     |   0.0289   |   0.0276   |     0.8027     |
| False         |        512       |     10     |   0.0269   |   0.0259   |     0.8194     |

### Comment
Fintuning result is remarkable and stunning. But just using a BERT output(wihtout fintuning) and put it through a single linear layer is not enought to handle data.

### Reference

My pytorch implementation is highly impressed by other works. Please check below to see other works.
1. https://github.com/huggingface/pytorch-transformers
2. https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784
