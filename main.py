# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = True           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)


    n_epochs = 50
    train_val_split = .7
    batch_size = 100
    hidden_dim = 20
    n_layers = 1
    lr = 0.01

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords('data_trainval.p')

    
    # Séparation de l'ensemble de données (entraînement et validation)
    n_train_dataset = int(len(dataset)*train_val_split)
    n_val_dataset = len(dataset)-n_train_dataset
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train_dataset, n_val_dataset])
   

    # Instanciation des dataloaders
    dataload_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)


    # Instanciation du model
    model = trajectory2seq(hidden_dim=hidden_dim, \
                         n_layers=n_layers, device=device, symb2int=dataset.symb2int, \
                         int2symb=dataset.int2symb, dict_size=dataset.dict_size, maxlen=dataset.max_len)
    nb_parameters = sum(p.numel() for p in model.parameters())
    print('\nNumber of parameters in the model : ', nb_parameters)
    # Initialisation des variables
    # À compléter

    if trainning:

        # Initialisation affichage
        if learning_curves:
            train_dist = []  # Historique des distances
            train_loss = []  # Historique des coûts
            val_dist = []
            val_loss = []
            fig, ax = plt.subplots(1)  # Initialisation figure

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=2)  # ignorer les symboles <pad>
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            dist = 0
            for batch_idx, data in enumerate(dataload_train):
                # Formatage des données
                data_seq, target_seq = data
                data_seq = data_seq.to(device).float()
                target_seq = target_seq.to(device).long()

                optimizer.zero_grad()  # Mise a zero du gradient
                output, hidden, attn = model(data_seq)  # Passage avant
                loss = criterion(output.view((-1, model.dict_size)), target_seq.view(-1))
                #loss = criterion(output, target_seq)

                loss.backward()  # calcul du gradient
                optimizer.step()  # Mise a jour des poids
                running_loss_train += loss.item()

                # calcul de la distance d'édition
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()
                M = len(output_list)
                for i in range(batch_size):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1)  # longueur mot a
                    Mb = b.index(1) if 1 in b else len(b)  # longueur mot b
                    dist += edit_distance(a[:Ma], b[:Mb]) / batch_size

            # Affichage pendant l'entraînement
            print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                epoch, n_epochs, (batch_idx + 1) * batch_size, len(dataload_train.dataset),
                                 100. * (batch_idx + 1) * batch_size / len(dataload_train.dataset),
                                 running_loss_train / (batch_idx + 1),
                                 dist / len(dataload_train)))

            # Validation
            running_loss_val = 0
            dist_val = 0
            model.eval()
            for batch_idx_val, data in enumerate(dataload_val):
                # Formatage des données
                data_seq, target_seq = data
                data_seq = data_seq.to(device).float()
                target_seq = target_seq.to(device).long()

                output, hidden, attn = model(data_seq)  # Passage avant
                loss = criterion(output.view((-1, model.dict_size)), target_seq.view(-1))

                running_loss_val += loss.item()

                # calcul de la distance d'édition
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()
                M = len(output_list)
                for i in range(batch_size):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1)  # longueur mot a
                    Mb = b.index(1) if 1 in b else len(b)  # longueur mot b
                    dist_val += edit_distance(a[:Ma], b[:Mb]) / batch_size

            # Affichage pendant la validation
            print(
                'Valid - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}\n'.format(
                    epoch, n_epochs, batch_idx_val * batch_size, len(dataload_val.dataset),
                                     100. * batch_idx_val * batch_size / len(dataload_val.dataset),
                                     running_loss_val / (batch_idx_val + 1),
                                     dist_val / len(dataload_val)))

            # Affichage graphique
            if learning_curves:
                train_loss.append(running_loss_train / len(dataload_train))
                train_dist.append(dist / len(dataload_train))
                val_loss.append(running_loss_val / len(dataload_val))
                val_dist.append(dist_val / len(dataload_val))
                ax.cla()
                ax.plot(train_loss, label='training loss')
                ax.plot(train_dist, label='training distance')
                ax.plot(val_loss, label='validation loss')
                ax.plot(val_dist, label='validation distance')
                ax.legend()
                plt.draw()
                plt.pause(0.01)

            # Enregistrer les poids
            torch.save(model, 'model.pt')

            # Terminer l'affichage d'entraînement
            


            # Ajouter les loss aux listes
            # À compléter

            # Enregistrer les poids
            # À compléter


            # Affichage
            if learning_curves:
                # visualization
                # À compléter
                pass

    if test:
        # Évaluation
        # À compléter

        # Charger les données de tests
        # À compléter

        # Affichage de l'attention
        # À compléter (si nécessaire)

        # Affichage des résultats de test
        # À compléter
        
        # Affichage de la matrice de confusion
        # À compléter

        # Évaluation

        # Chargement des poids
        model = torch.load('model.pt')
        dataset.symb2int = model.symb2int
        dataset.int2symb = model.int2symb

        # Affichage des résultats
        for i in range(10):
            # Extraction d'une séquence du dataset de validation
            fr_seq, target_seq = dataset[np.random.randint(0, len(dataset))]

            # Évaluation de la séquence
            output, hidden, attn = model(torch.tensor(fr_seq)[None,:].to(device).float())
            out = torch.argmax(output, dim=2).detach().cpu()[0, :].tolist()

            # Affichage
            #in_seq = [model.int2symb[i] for i in fr_seq.detach().cpu().tolist()]
            target = [model.int2symb[i] for i in target_seq.detach().cpu().tolist()]
            out_seq = [model.int2symb[i] for i in out]

            out_seq = out_seq[:out_seq.index('<eos>') + 1]
            #in_seq = in_seq[:in_seq.index('<eos>') + 1]
            target = target[:target.index('<eos>') + 1]

            #print('Input:  ', ' '.join(in_seq))
            print('Target: ', ' '.join(target))
            print('Output: ', ' '.join(out_seq))
            print('')
            #if display_attention:
            #    attn = attn.detach().cpu()[0, :, :]
            #    plt.figure()
            #    plt.imshow(attn[0:len(in_seq), 0:len(out_seq)], origin='lower', vmax=1, vmin=0, cmap='pink')
            #    plt.xticks(np.arange(len(out_seq)), out_seq, rotation=45)
            #    plt.yticks(np.arange(len(in_seq)), in_seq)
            #    plt.show()