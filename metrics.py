# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np

def edit_distance(x,y):
    # Calcul de la distance d'édition
    D = np.zeros([len(x)+1, len(y)+1])
    for i in range(0, len(x)):
        D[i][0] = i
    for j in range(0, len(y)):
        D[0][j] = j

    for i in range(1, len(x)):
        for j in range(1, len(y)):
            min = np.minimum(D[i-1][j]+1, D[i][j-1]+1)
            if x[i] == y[j]:
                D[i][j] = np.minimum(min, D[i-1][j-1])
            else:
                D[i][j] = np.minimum(min, D[i - 1][j - 1]+1)

    distance = D[len(x)-1][len(y)-1]

    return distance

def confusion_matrix(true, pred, ignore=[]):
    # Calcul de la matrice de confusion
    K = 28  # Nombre de symboles
    result = np.zeros((K, K))

    for i in range(len(true)):
        result[true[i]-1][pred[i]-1] += 1

    # Calcul des nombres de vrai et faux positif et vrai et faux négatif
    # TP = np.diag(result)
    # FP = result.sum(axis=0) - np.diag(result)
    # FN = result.sum(axis=1) - np.diag(result)
    # TN = result.sum() - (FP + FN + TP)

    return result
