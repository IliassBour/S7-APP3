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

        self.gru_coord = nn.GRU(2, self.hidden_dim, self.n_layers, batch_first=True)
        #self.gru_coord = nn.GRU(self.maxlen['coord'], self.hidden_dim, self.n_layers, batch_first=True)
        self.gru_word = nn.GRU(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)
        #self.bi_gru_coord = nn.GRU(2, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True)
        #self.bi_gru_word = nn.GRU(self.hidden_dim, self.hidden_dim, 2*self.n_layers, batch_first=True, bidirectional=False)


        # Couches pour attention
        self.att_combine = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.hidden2query = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Couche dense pour la sortie
        self.fc = nn.Linear(self.hidden_dim, self.dict_size)
        self.to(device)
        # À compléter

    def encoder(self, x):
        # Encodeur

        # Sans bidirectionnelle
        out, hidden = self.gru_coord(x)
        # Avec bidirectionnelle
        #out, hidden = self.bi_gru_coord(x)

        return out, hidden

    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.maxlen['word']  # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1]  # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, max_len, self.dict_size)).to(self.device)  # Vecteur de sortie du décodage

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            embed = self.word_embedding(vec_in)
            #Sans bidirectionnelle
            out, hidden = self.gru_word(embed, hidden)
            #Avec bidirectionnelle
            #out, hidden = self.bi_gru_word(embed, hidden)

            out = self.fc(out)
            vec_in = torch.argmax(out, axis=2)
            vec_out[:, i:i + 1, :] = out
        return vec_out, hidden, None

    def attentionModule(self, query, values):
        # Module d'attention
        # Couche dense à l'entrée du module d'attention
        query = self.hidden2query(query)
        # Attention
        #query = torch.reshape(query, (query.shape[0], query.shape[2], query.shape[1]))
        query = torch.transpose(query, 1, 2)
        attention_weights = torch.softmax(torch.bmm(values, query), dim=1)
        #tmp = torch.reshape(attention_weights, (attention_weights.shape[0], attention_weights.shape[2], attention_weights.shape[1]))
        tmp = torch.transpose(attention_weights, 1, 2)
        attention_output = torch.bmm(tmp, values)

        return attention_output, attention_weights

    def decoderWithAttn(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.maxlen['word']  # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1]  # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, max_len, self.dict_size)).to(self.device)  # Vecteur de sortie du décodage
        attention_weights = torch.zeros([batch_size, self.maxlen['coord'], max_len]).to(self.device)

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            embed = self.word_embedding(vec_in)
            out, hidden = self.gru_word(embed, hidden)
            attention_output, attention_weights[:, :, i:i + 1] = self.attentionModule(out, encoder_outs)
            out = self.att_combine(torch.cat([out, attention_output], dim=2))
            out = self.fc(out)
            vec_in = torch.argmax(out, axis=2)
            vec_out[:, i:i + 1, :] = out
        return vec_out, hidden, attention_weights

    def forward(self, x):
        out, h = self.encoder(x)
        #out, hidden, attn = self.decoder(out, h)
        out, hidden, attn = self.decoderWithAttn(out, h)
        return out, hidden, attn
    

