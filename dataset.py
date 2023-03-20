import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle


class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'

        start_symbol_coord = np.array([[-1], [-1]])
        stop_symbol_coord = np.array([[-2], [-2]])
        pad_symbol_coord = np.array([[-3], [-3]])

        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)

        # Extraction des symboles
        self.symb2int = {start_symbol: 0, stop_symbol: 1, pad_symbol: 2, 'a':3, 'b':4, 'c':5, 'd':6, 'e':7, 'f':8, 'g':9, 'h':10, 'i':11, 'j':12, 'k':13, 'l':14, 'm':15, 'n':16, 'o':17, 'p':18, 'q':19, 'r':20, 's':21, 't':22, 'u':23, 'v':24, 'w':25, 'x':26, 'y':27, 'z':28}
        self.int2symb = {v: k for k, v in self.symb2int.items()}
        # self.max_len = max(map(len, self.seq_list)) + 1
        self.max_len = dict()
        self.max_len['coord'] = 458
        self.max_len['word'] = 6
        #self.max_len_coord = 457 + 1
        #self.max_len_word = 6
        # À compléter


        # Ajout du padding aux séquences
        for word in self.data:
            if word[1].shape[1] < self.max_len['coord']:
                for i in range(self.max_len['coord'] - word[1].shape[1]):
                    if i == 0:
                        word[1] = np.append(word[1], stop_symbol_coord, axis=1)
                    else:
                        word[1] = np.append(word[1], pad_symbol_coord, axis=1)
            if len(word[0]) < self.max_len['word']:
                word[0] = list(word[0])
                for i in range(self.max_len['word'] - len(word[0])):
                    if i == 0:
                        word[0].append(stop_symbol)
                    else:
                        word[0].append(pad_symbol)

        self.dict_size = {'coord':self.__len__(), 'word':len(self.int2symb)}
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word = self.data[idx][0]
        coord = self.data[idx][1]

        data_seq = coord
        target_seq = [self.symb2int[j] for j in word]
        return torch.tensor(data_seq), torch.tensor(target_seq)

    def visualisation(self, idx):
        # Visualisation des échantillons
        # À compléter (optionel)
        pass
        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))
