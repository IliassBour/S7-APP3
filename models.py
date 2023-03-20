# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen

        # Definition des couches
        # Couches pour rnn
        self.word_embedding = nn.Embedding(self.dict_size, self.hidden_dim)

        #self.gru_coord = nn.GRU(2, self.hidden_dim, self.n_layers, batch_first=True)
        self.gru_coord = nn.GRU(self.maxlen['coord'], self.hidden_dim, self.n_layers, batch_first=True)
        self.gru_word = nn.GRU(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)
        # À compléter

        # Couches pour attention
        # À compléter

        # Couche dense pour la sortie
        self.fc = nn.Linear(self.hidden_dim, self.dict_size)
        self.to(device)
        # À compléter

    def encoder(self, x):
        # Encodeur
        #embed = self.coord_embedding(x)
        #test = x.view(-1, x.size(2))
        #embed = self.coord_embedding(x.view(-1, x.size(2)))

        out, hidden = self.gru_coord(x)
        #out, hidden = self.lstm(x)
        #out, hidden = self.gru(x)

        #out = self.coord_word_lin(out)

        return out, hidden

    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.maxlen['word']  # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1]  # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, max_len, self.dict_size)).to(
            self.device)  # Vecteur de sortie du décodage

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            embed = self.word_embedding(vec_in)
            out, hidden = self.gru_word(embed, hidden)

            out = self.fc(out)
            vec_in = torch.argmax(out, axis=2)
            vec_out[:, i:i + 1, :] = out
        return vec_out, hidden, None

    def forward(self, x):
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out, h)
        return out, hidden, attn
    

