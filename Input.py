from torch.utils.data import Dataset, DataLoader
import gensim.models
import torch
import numpy as np
input_shape = 300
class word2vec_data(Dataset):
    def __init__(self, file):
        super(word2vec_data, self).__init__()
        w2v = gensim.models.KeyedVectors.load_word2vec_format('cn.cbow.bin', binary=True, unicode_errors='ignore')
        #w2v = gensim.models.word2vec.Word2Vec.load('Model')
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.length = len(lines)

        self.sentence_length = 0
        for i, line in enumerate(lines):
            words = line.split()[10:]
            # print(words)
            i = 0
            while (i != len(words)):
                if not words[i] in w2v:
                    # print(words[i])
                    words.pop(i)
                    i -= 1
                i += 1
            if len(words) > self.sentence_length:
                self.sentence_length = len(words)
        print('sentence length:', self.sentence_length)

        self.dataset = []
        for line in lines:
            words = line.split()[10:]
            #print(words)
            i = 0
            while(i != len(words)):
                if not words[i] in w2v:
                    #print(words[i])
                    words.pop(i)
                    i -= 1
                i += 1
            data = np.zeros((1, self.sentence_length, input_shape), dtype=np.float)
            for i, word in enumerate(words):
                data[0, i, :] = w2v[word]
            data = torch.tensor(data, dtype=torch.float)
            #print(data.shape)

            labels = line.split()[2:10]
            total = int(line.split()[1].split(':')[1])
            label1 = np.zeros((8, ), dtype=np.float)
            max_num = 0
            label2 = 0
            for i, label in enumerate(labels):
                num = int(label.split(':')[1])
                label1[i] = num / (total + 0.0)
                if num >= max_num:
                    max_num = num
                    label2 = i
            label1 = torch.tensor(label1, dtype=torch.float)

            self.dataset.append({
                'input': data,
                'label1':label1,
                'label2':torch.tensor([label2], dtype=torch.int)
            })


    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.dataset[item]
