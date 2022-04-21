
import cv2
import os, sys
import glob

import math
import numpy as np

from matplotlib import pyplot as plt

from random import seed
from random import random

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
# imagens recuperadas foram afetadas por essas transformações. Dica: você pode usar as rotinas
# getRotationMatrix2D e warpAffine do OpenCV.

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

def TestImage():
    # Reading the image
    image = cv2.imread(PATH_TO_IMGS+'bird02.pgm')
    final_image = Random_RotationScale(image)

    cv2.imshow("final image:", final_image)
    # cv2.imwrite('final_image.pgm', rotated_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def Random_RotationScale(original_img):
    scale = 0.85 + (random() * (1.15 - 0.85)) # [0.85, 1.15]
    rotation = random() * 360 # [0, 360 graus]
    print("Scale: ",scale, " --- Rotation: ", rotation, "degrees")

    # Extracting height and width from image shape
    height, width = original_img.shape[:2]
  
    # get the center coordinates of the image to create the 2D rotation matrix
    center = (width/2, height/2)
    
    # using cv2.getRotationMatrix2D() to get the rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rotation, scale=scale)
  
    # para evitar corte no limite da imagem
    radians = math.radians(rotation)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotate_matrix[0, 2] += ((bound_w / 2) - center[0])
    rotate_matrix[1, 2] += ((bound_h / 2) - center[1])

    # rotate the image using cv2.warpAffine rotation degree anticlockwise
    rotated_image = cv2.warpAffine(src=original_img, M=rotate_matrix, dsize=(bound_w, bound_h))
  
    return rotated_image


if __name__ == "__main__":

    seed(1)

    # print("teste")
    # CBIR().openDir()
    TestImage()