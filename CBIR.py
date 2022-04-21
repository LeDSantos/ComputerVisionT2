
import cv2
import os, sys
import glob

import math
import numpy as np

from matplotlib import pyplot as plt

# Enunciado: o problema de Content-based image retrieval (CBIR) consiste em fornecer uma
# imagem de consulta e recuperar imagens semelhantes em uma base de dados. O conceito de
# semelhante depende da aplicação, normalmente envolvendo informação de cor, textura e forma.
# Neste trabalho, o objetivo é criar um sistema de CBIR usando informação de forma como
# medida de similaridade. Cada dupla deve desenvolver um sistema simples que receba como entrada
# uma imagem de busca, um número inteiro positivo N, e então retornar as N imagens da base de
# dados mais semelhantes à imagem de busca, em ordem decrescente de similaridade. Se a imagem
# de busca estiver na base de dados, idealmente ela deve ser a primeira imagem a ser recuperada.

# CMP 197 apenas: escreva uma rotina que aplique uma variação aleatória de escala (com um
# fator selecionado no intervalo [0.85, 1.15]) e uma rotação aleatória tanto na imagem de consulta
# quanto nas imagens a serem buscadas. Para uma mesma imagem de consulta, avalie o quanto as
# imagens recup eradas foram afetadas p or essas transformações. Dica: você pode usar as rotinas
# getRotationMatrix2D e warpAffine do Op enCV.

PATH_TO_IMGS = "./images/"

class CBIR:
    def __init__(self):

        files = self.openDir()

        for img_file in files:
            # Carrega imagem
            img = cv2.imread(img_file)
            print("Usando imagem: "+img_file)
            img_file_out=img_file.split("/")[-1]
            # cv2.imshow('Imagem '+img_file_out, img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
        
        print("CBIR executado")


    def openDir(self):
        files = glob.glob(PATH_TO_IMGS+"*.pgm")
        # print(files)
        return files

if __name__ == "__main__":

    # print("teste")
    CBIR().openDir()