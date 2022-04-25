import glob
from itertools import count
import cv2
import numpy as np
import operator
import sys
from scipy.spatial import distance
import re
import math
from random import seed
from random import random

from skimage.feature import hog
from skimage import exposure

PATH_TO_IMGS = "./images/"
THRESHOLD_QUERY = 5.5


class OpenFile:
    def search_file(self):
        files = glob.glob(PATH_TO_IMGS+"*.pgm")
        return files
    
    def process_search_img(self, input_img_path):

        img_gray = cv2.imread(input_img_path, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        threshold = cv2.bitwise_not(threshold)

        shape_x, shape_y = threshold.shape

        #DESCRITOR HOG
        fd_q, hog_image_query = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=False)

        #GAUSSIAN BLUR PARA CONSIDERAÇÃO DE AUMENTO NA IMAGEM
        img_blur1_input = cv2.GaussianBlur(img_gray,(1,1),0)
        img_blur3_input = cv2.GaussianBlur(img_gray,(3,3),0)
        img_blur5_input = cv2.GaussianBlur(img_gray,(5,5),0)

        #ROTAÇÃO NA IMAGEM
        rotated_imgs = []
        for ind in range(45,360,45):
            rotaded = self.Rotation(img_gray,ind)
            rotated_imgs.append(rotaded)
        
        #DETECÇÃO DE BORDAS COM CANNY
        canny_input = cv2.Canny(img_blur3_input,225,250)

        return (shape_x, shape_y, fd_q, img_blur1_input, img_blur3_input, img_blur5_input, rotated_imgs, canny_input)

    def process_search_img_MOD(self, input_img_path):

        img_gray = cv2.imread(input_img_path, cv2.COLOR_BGR2GRAY)
        img_gray = self.Random_RotationScale(img_gray)
        
        ret, threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        threshold = cv2.bitwise_not(threshold)

        shape_x, shape_y = threshold.shape

        #DESCRITOR HOG
        fd_q, hog_image_query = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=False)

        #GAUSSIAN BLUR PARA CONSIDERAÇÃO DE AUMENTO NA IMAGEM
        img_blur1_input = cv2.GaussianBlur(img_gray,(1,1),0)
        img_blur3_input = cv2.GaussianBlur(img_gray,(3,3),0)        
        img_blur5_input = cv2.GaussianBlur(img_gray,(5,5),0)

        #ROTAÇÃO NA IMAGEM
        rotated_imgs = []
        for ind in range(45,360,45):
            rotaded = self.Rotation(img_gray,ind)
            rotated_imgs.append(rotaded)
 
        #DETECÇÃO DE BORDAS COM CANNY
        canny_input = cv2.Canny(img_blur3_input,225,250)

        return (shape_x, shape_y, fd_q, img_blur1_input, img_blur3_input, img_blur5_input, rotated_imgs, canny_input)

    def make_descriptors(self, input_img_path, shape_x, shape_y, fd_q, img_blur1_input, img_blur3_input, img_blur5_input, rotated_imgs, canny_input, input_img_test_path):
        img_tst = cv2.imread(input_img_test_path, cv2.COLOR_BGR2GRAY)
        ret, img_tst = cv2.threshold(img_tst, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        x,y = shape_x, shape_y
        img_tst = cv2.resize(img_tst,(y,x))

        #DESCRITOR HOG
        fd_db, hog_image_database = hog(img_tst, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=False)

        #DISTÂNCIA EUCLIDIANA
        eucli_fd = self.euclidean_1D(fd_q,fd_db)

        #ERRO MÉDIO QUADRÁTICO
        mse_fd = self.mse_1d(fd_q,fd_db)

        #DISTÂNCIA COSSENO
        cos_fd = distance.cosine(fd_q,fd_db)

        #GAUSSIAN BLUR PARA CONSIDERAÇÃO DE AUMENTO NA IMAGEM
        erro1_mse = self.mse_2imgs(img_tst,img_blur1_input,input_img_path,"IMG_BLUR1")
        blur1_eucli = self.euclidean_2D(img_blur1_input,img_tst)
        cos_blur1 = distance.cosine(img_blur1_input.flatten(),img_tst.flatten())

        erro3_mse = self.mse_2imgs(img_tst,img_blur3_input,input_img_path,"IMG_BLUR3")
        blur3_eucli = self.euclidean_2D(img_blur3_input,img_tst)
        cos_blur3 = distance.cosine(img_blur1_input.flatten(),img_tst.flatten())
        
        erro5_mse = self.mse_2imgs(img_tst,img_blur5_input,input_img_path,"IMG_BLUR5")
        blur5_eucli = self.euclidean_2D(img_blur5_input,img_tst)
        cos_blur5 = distance.cosine(img_blur1_input.flatten(),img_tst.flatten())

        #ROTAÇÃO NA IMAGEM
        rotate_values = []
        for it in range(len(rotated_imgs)):
            rotaded = rotated_imgs[it]
            img_tst_resize = cv2.resize(img_tst,(rotaded.shape[0],rotaded.shape[1]))
            if rotaded.shape[0] != img_tst_resize.shape[0]:
                img_tst_resize = cv2.resize(img_tst,(rotaded.shape[1],rotaded.shape[0]))
            rotate_eucli = self.euclidean_2D(rotaded,img_tst_resize)
            rotate_values.append(rotate_eucli)
        
        rot45_eucli = rotate_values[0]
        rot90_eucli = rotate_values[1]
        rot135_eucli = rotate_values[2]
        rot180_eucli = rotate_values[3]
        rot225_eucli = rotate_values[4]
        rot270_eucli = rotate_values[5]
        rot315_eucli = rotate_values[6]
        
        #DETECÇÃO DE BORDAS COM CANNY
        img_blur3_db = cv2.GaussianBlur(img_tst,(3,3),0)
        canny_input = cv2.Canny(img_blur3_input,225,250)
        canny_db = cv2.Canny(img_blur3_db,225,250)
        canny_eucli = self.euclidean_2D(canny_input,canny_db)

        return (eucli_fd,input_img_test_path,mse_fd,cos_fd,blur1_eucli,blur3_eucli,blur5_eucli,rot45_eucli,rot90_eucli,rot135_eucli,
        rot180_eucli,rot225_eucli,rot270_eucli,rot315_eucli,canny_eucli,erro1_mse,erro3_mse,erro5_mse,
        cos_blur1,cos_blur3,cos_blur5)
    
    def make_descriptors_MOD(self, input_img_path, shape_x, shape_y, fd_q, img_blur1_input, img_blur3_input, img_blur5_input, rotated_imgs, canny_input, input_img_test_path):
        img_tst = cv2.imread(input_img_test_path, cv2.COLOR_BGR2GRAY)

        img_tst = self.Random_RotationScale(img_tst)

        ret, img_tst = cv2.threshold(img_tst, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        x,y = shape_x, shape_y
        img_tst = cv2.resize(img_tst,(y,x))

        #DESCRITOR HOG
        fd_db, hog_image_database = hog(img_tst, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=False)

        #DISTÂNCIA EUCLIDIANA
        eucli_fd = self.euclidean_1D(fd_q,fd_db)

        #ERRO MÉDIO QUADRÁTICO
        mse_fd = self.mse_1d(fd_q,fd_db)

        #DISTÂNCIA COSSENO
        cos_fd = distance.cosine(fd_q,fd_db)

        #GAUSSIAN BLUR PARA CONSIDERAÇÃO DE AUMENTO NA IMAGEM
        erro1_mse = self.mse_2imgs(img_tst,img_blur1_input,input_img_path,"IMG_BLUR1")
        blur1_eucli = self.euclidean_2D(img_blur1_input,img_tst)
        cos_blur1 = distance.cosine(img_blur1_input.flatten(),img_tst.flatten())

        erro3_mse = self.mse_2imgs(img_tst,img_blur3_input,input_img_path,"IMG_BLUR3")
        blur3_eucli = self.euclidean_2D(img_blur3_input,img_tst)
        cos_blur3 = distance.cosine(img_blur1_input.flatten(),img_tst.flatten())
        
        erro5_mse = self.mse_2imgs(img_tst,img_blur5_input,input_img_path,"IMG_BLUR5")
        blur5_eucli = self.euclidean_2D(img_blur5_input,img_tst)
        cos_blur5 = distance.cosine(img_blur1_input.flatten(),img_tst.flatten())

        #ROTAÇÃO NA IMAGEM
        rotate_values = []
        for ind in range(len(rotated_imgs)):#range(45,360,45):
            rotaded = rotated_imgs[ind] #self.Rotation(img_gray,ind)
            img_tst_resize = cv2.resize(img_tst,(rotaded.shape[0],rotaded.shape[1]))
            if rotaded.shape[0] != img_tst_resize.shape[0]:
                img_tst_resize = cv2.resize(img_tst,(rotaded.shape[1],rotaded.shape[0]))
            rotate_eucli = self.euclidean_2D(rotaded,img_tst_resize)
            rotate_values.append(rotate_eucli)
        
        rot45_eucli = rotate_values[0]
        rot90_eucli = rotate_values[1]
        rot135_eucli = rotate_values[2]
        rot180_eucli = rotate_values[3]
        rot225_eucli = rotate_values[4]
        rot270_eucli = rotate_values[5]
        rot315_eucli = rotate_values[6]
        

        #DETECÇÃO DE BORDAS COM CANNY
        img_blur3_db = cv2.GaussianBlur(img_tst,(3,3),0)
        canny_input = cv2.Canny(img_blur3_input,225,250)
        canny_db = cv2.Canny(img_blur3_db,225,250)
        canny_eucli = self.euclidean_2D(canny_input,canny_db)

        return (eucli_fd,input_img_test_path,mse_fd,cos_fd,blur1_eucli,blur3_eucli,blur5_eucli,rot45_eucli,rot90_eucli,rot135_eucli,
        rot180_eucli,rot225_eucli,rot270_eucli,rot315_eucli,canny_eucli,erro1_mse,erro3_mse,erro5_mse,
        cos_blur1,cos_blur3,cos_blur5)

    def mse_2imgs(self,imgA,imgB,query_name,prefix):
        erro = np.sum((imgA.astype("float") - imgB.astype("float")) **2)
        erro /= float(imgA.shape[0] * imgA.shape[1])
        return erro

    def mse_1d(self,imgA,imgB):
        mse_1d = np.sum((imgA-imgB)**2)
        mse_1d /= len(imgA)
        return mse_1d

    def euclidean_1D(self,imgA,imgB):
        eucli_1d = 0
        for xi in range(0,len(imgA)):
            eucli_1d += np.square(imgA[xi] - imgB[xi])
        eucli_1d = np.sqrt(eucli_1d)
        return eucli_1d

    def euclidean_2D(self,imgA,imgB):
        x,y = imgA.shape
        eucli_sum = 0
        for xi in range(0,x-1):
            for yi in range(0,y-1):
                eucli_sum = eucli_sum + np.square((int(imgA[xi][yi]) - int(imgB[xi][yi])))
        eucli = np.sqrt(eucli_sum)
        return eucli

    def find_related_imgs(self, query, shape_x, shape_y, fd_q, img_blur1_input, img_blur3_input, img_blur5_input, rotated_imgs, canny_input,imgv):
        similar_fd = []
        for img in imgv:

            [eucli_fd,img_path,mse_fd,cos_fd,blur1_eucli,blur3_eucli,blur5_eucli,
            rot45_eucli,rot90_eucli,rot135_eucli,rot180_eucli,rot225_eucli,rot270_eucli,
            rot315_eucli,canny_eucli,erro1_mse,erro3_mse,erro5_mse,cos_blur1,cos_blur3,cos_blur5] = of.make_descriptors(query, shape_x, shape_y, fd_q, img_blur1_input, img_blur3_input, img_blur5_input, rotated_imgs, canny_input,img)
            
            similar_fd.append((img_path,eucli_fd,mse_fd,cos_fd,blur1_eucli,blur3_eucli,blur5_eucli,rot45_eucli,
            rot90_eucli,rot135_eucli,rot180_eucli,rot225_eucli,rot270_eucli,rot315_eucli,canny_eucli,erro1_mse,
            erro3_mse,erro5_mse,cos_blur1,cos_blur3,cos_blur5))
        return similar_fd
    
    def find_related_imgs_MOD(self, query, shape_x, shape_y, fd_q, img_blur1_input, img_blur3_input, img_blur5_input, rotated_imgs, canny_input,imgv):
        similar_fd = []
        for img in imgv:
            [eucli_fd,img_path,mse_fd,cos_fd,blur1_eucli,blur3_eucli,blur5_eucli,
            rot45_eucli,rot90_eucli,rot135_eucli,rot180_eucli,rot225_eucli,rot270_eucli,
            rot315_eucli,canny_eucli,erro1_mse,erro3_mse,erro5_mse,cos_blur1,cos_blur3,cos_blur5] = of.make_descriptors_MOD(query, shape_x, shape_y, fd_q, img_blur1_input, img_blur3_input, img_blur5_input, rotated_imgs, canny_input,img)
            
            similar_fd.append((img_path,eucli_fd,mse_fd,cos_fd,blur1_eucli,blur3_eucli,blur5_eucli,rot45_eucli,
            rot90_eucli,rot135_eucli,rot180_eucli,rot225_eucli,rot270_eucli,rot315_eucli,canny_eucli,erro1_mse,
            erro3_mse,erro5_mse,cos_blur1,cos_blur3,cos_blur5))
        return similar_fd

    def show_related_img(self,query,similar_fd,N,save_result):
        img = []
        open_img = cv2.imread(query,cv2.COLOR_BGR2GRAY)
        ximg,yimg = open_img.shape
        for x in range(0,N):
            img.append(cv2.resize(cv2.imread(similar_fd[x][0],cv2.COLOR_BGR2GRAY),(yimg,ximg)))
        if save_result:
            # cv2.imwrite("./results/teste_bird.jpg",np.concatenate(img,axis=1))
            return np.concatenate(img,axis=1)

    def precision_recall(self,N,similar_fd,name_input):
        cont = 0
        for x in range(0,N):
            if re.sub(r'[^a-zA-Z]', '', name_input) in similar_fd[x][0]:
                cont+=1
        P = cont / N
        R = cont / 12
        F = 0
        if P > 0 or R > 0:
            F = (2* P * R) / (P+R)
        return P,R,F

    def Rotation(self,original_img,rotation):
        # Extracting height and width from image shape
        height, width = original_img.shape[:2]
    
        # get the center coordinates of the image to create the 2D rotation matrix
        center = (width/2, height/2)
        
        # using cv2.getRotationMatrix2D() to get the rotation matrix
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rotation, scale=1)
    
        # para evitar corte no limite da imagem
        radians = math.radians(rotation)
        sin = math.sin(radians)
        cos = math.cos(radians)
        bound_w = int((height * abs(sin)) + (width * abs(cos)))
        bound_h = int((height * abs(cos)) + (width * abs(sin)))

        rotate_matrix[0, 2] += ((bound_w / 2) - center[0])
        rotate_matrix[1, 2] += ((bound_h / 2) - center[1])

        # rotate the image using cv2.warpAffine rotation degree anticlockwise
        rotated_image = cv2.warpAffine(src=original_img, M=rotate_matrix, dsize=(bound_w, bound_h), borderValue=(255,255,255))
  
        return rotated_image
    
    def Random_RotationScale(self,original_img):
        scale = 0.85 + (random() * (1.15 - 0.85)) # [0.85, 1.15]
        rotation = random() * 360 # [0, 360 graus]

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
        rotated_image = cv2.warpAffine(src=original_img, M=rotate_matrix, dsize=(bound_w, bound_h), borderValue=(255,255,255))
    
        return rotated_image

    def evaluate_results(self,similar_fd,N,input_name):
        P_VALUE = []
        max_ind = []
        for x in range(1,21):
            similar_fd.sort(key = operator.itemgetter(x)) 
            P,R,F = of.precision_recall(int(N),similar_fd,input_name)
            print("P: " + str(P) + " | R: " + str(R) + " | F: " + str(F))
            P_VALUE.append(P)
        for x in range(0,len(P_VALUE)):
            if np.max(P_VALUE) == P_VALUE[x]:
                print("MAX P_VALUE: "+str(np.max(P_VALUE))+" INDICE: "+str(x))
                max_ind.append(x+1)
        if len(max_ind) > 1:
            img = []
            for x in range(0,len(max_ind)):
                similar_fd.sort(key = operator.itemgetter(max_ind[x]))
                img.append(self.show_related_img(PATH_TO_IMGS+str(sys.argv[1])+".pgm",similar_fd,int(N),1))
            cv2.imshow("RESULTADOS",np.concatenate(img,axis=0))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # cv2.imwrite("./results/result_mod2.jpg",np.concatenate(img,axis=0))
        else:
            similar_fd.sort(key = operator.itemgetter(np.argmax(P_VALUE)+1))
            img = self.show_related_img(PATH_TO_IMGS+str(sys.argv[1])+".pgm",similar_fd,int(N),1)
            cv2.imshow("RESULTADOS",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # cv2.imwrite("./results/result_mod2.jpg",img)
        return similar_fd
    
    def CBIR(self,input_img,N):
        
        img_files = self.search_file()
        
        query = input_img
        
        (shape_x, shape_y, fd_q, img_blur1_input, img_blur3_input, img_blur5_input, rotated_imgs, canny_input) = self.process_search_img(input_img)

        similar_fd = []
        similar_fd = self.find_related_imgs(query, shape_x, shape_y, fd_q, img_blur1_input, img_blur3_input, img_blur5_input, rotated_imgs, canny_input,img_files)

        similar_fd = self.evaluate_results(similar_fd,N,sys.argv[1])
        save_result = sys.argv[3]
        self.show_related_img(query,similar_fd,int(N),int(save_result))

    def CBIR_MOD(self,input_img,N):
        img_files = self.search_file()
        
        query = input_img
           
        (shape_x, shape_y, fd_q, img_blur1_input, img_blur3_input, img_blur5_input, rotated_imgs, canny_input) = self.process_search_img_MOD(input_img)
    
        similar_fd = []
        similar_fd = self.find_related_imgs_MOD(query, shape_x, shape_y, fd_q, img_blur1_input, img_blur3_input, img_blur5_input, rotated_imgs, canny_input,img_files)

        similar_fd = self.evaluate_results(similar_fd,N,sys.argv[1])
        save_result = sys.argv[3]
        self.show_related_img(query,similar_fd,int(N),int(save_result))

""""
=================ARGUMENTOSS PARA RODAR O PROGRAMA=================
    python3 script.py argv[1] argv[2] argv[3]
    - argv[1] = SEARCH_IMG (Somente o nome do arquivo de entrada sem a extensão)
    - argv[2] = N (Quantidade de imagens a serem retornadas)
    - argv[3] = SAVE_RESULT (0 - Não salva a imagem com os correspondentes | 1 - Salva a imagem com os correpondentes)

    função CBIR - faz a busca das N imagens mais semelheantes com a imagem de entrada
    função CBIR_MOD - Aplica uma escala e rotação, tanto na imagem de entrada quanto nas imagens de busca,
    faz a busca das N imagens mais semelheantes com a imagem de entrada
"""

if __name__ == "__main__":
    of = OpenFile()

    seed(2)

    # of.CBIR(PATH_TO_IMGS+str(sys.argv[1])+".pgm",sys.argv[2])
    of.CBIR_MOD(PATH_TO_IMGS+str(sys.argv[1])+".pgm",sys.argv[2])
