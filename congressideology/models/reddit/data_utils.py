import torch.utils.data
from sklearn.preprocessing import LabelEncoder

from congressideology.config import reddit_config


class Lang:
    def __init__(self, min_count):
        """Language class

        Parameters
        ----------
        name : str
            Language name.
        """
        self.min_count = min_count
        self.word2index = {"PAD": reddit_config.PAD_token,
                           "UNK": reddit_config.UNK_token}
        self.word2count = {}
        self.index2word = {reddit_config.PAD_token: "PAD",
                           reddit_config.UNK_token: "UNK"}
        self.n_words = 2
        self.embeddings = None
        self.pretrained_inds = []

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        # add word to the language
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def trim(self):
        # trim vocabulary
        keep_words = []
        for word, count in self.word2count.items():
            if count >= self.min_count:
                keep_words.append((word, count))

        self.word2index = {"PAD": reddit_config.PAD_token,
                           "UNK": reddit_config.UNK_token}
        self.word2count = {}
        self.index2word = {reddit_config.PAD_token: "PAD",
                           reddit_config.UNK_token: "UNK"}
        self.n_words = 2

        for word, count in keep_words:
            self.addWord(word)
            self.word2count[word] = count


def prepare_data(comments, subreddits, min_count, label_encoder=None):
    if label_encoder is not None:
        subreddit_indices = label_encoder.transform(subreddits)
    else:
        label_encoder = LabelEncoder()
        subreddit_indices = label_encoder.fit_transform(subreddits)
    pairs = list(zip(comments, subreddit_indices))
    lang = Lang(min_count)
    for pair in pairs:
        lang.addSentence(pair[0])

    lang.trim()

    return {'lang': lang, 'pairs': pairs, 'label_encoder': label_encoder}


def indexesFromSentence(lang, sentence):
    # convert tokens to indices
    tokens = sentence[:reddit_config.max_length]
    indices = [lang.word2index.get(word, reddit_config.UNK_token)
               for word in tokens]
    return indices


class RedditDataset(torch.utils.data.Dataset):
    def __init__(self, lang, pairs):
        self.lang = lang
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_indices = indexesFromSentence(self.lang, self.pairs[idx][0])
        target_class = self.pairs[idx][1]

        return input_indices[0:reddit_config.max_length], target_class


def pad_seq(seq, max_length):
    # pad sequences
    seq += [reddit_config.PAD_token for i in range(max_length - len(seq))]
    return seq


def text_collate_func(batch):
    seq_pairs = sorted(batch,
                       key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_classes = zip(*seq_pairs)

    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(seq, max(input_lengths)) for seq in input_seqs]

    return {'input': torch.LongTensor(input_padded).to(reddit_config.device),
            'input_length': torch.LongTensor(input_lengths).to(reddit_config.device),
            'target': torch.LongTensor(target_classes).to(reddit_config.device)}
