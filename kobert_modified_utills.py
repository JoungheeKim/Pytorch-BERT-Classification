import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from gluonnlp.data import SentencepieceTokenizer
import gluonnlp
import collections


## 맞춤형 gluonnlp를 쓰지 않고 pytorch Tokenizer
class KoBertTokenizer(object):
    def __init__(self, basic_tokenizer, vocab, max_len=None):
        ## basic_tokenizer must be gluonnlp.data.transforms.SentencepieceTokenizer
        ## vocab must be gluonnlp.vocab.bert.BERTVocab
        if type(basic_tokenizer) is not gluonnlp.data.transforms.SentencepieceTokenizer:
            raise ValueError("basic_tokenizer must be gluonnlp.data.transforms.SentencepieceTokenizer")
        if type(vocab) is not gluonnlp.vocab.bert.BERTVocab:
            raise ValueError("vocab must be gluonnlp.vocab.bert.BERTVocab")
        self.vocab = vocab
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for ids, tok in enumerate(vocab.idx_to_token)])
        self.basic_tokenizer = basic_tokenizer
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer(text):
            split_tokens.append(token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

def get_kobert_model_and_tokenizer():
    tok_path = get_tokenizer()
    basic_tokenizer = SentencepieceTokenizer(tok_path)
    bert_base, vocab = get_pytorch_kobert_model()
    kobert_tokenizer = KoBertTokenizer(basic_tokenizer, vocab)

    return bert_base, kobert_tokenizer