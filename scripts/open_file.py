import glob
from itertools import count
import cv2
import numpy as np
import imutils
import operator
import sys
from scipy.spatial import distance

from skimage.feature import hog
from skimage import exposure

PATH_TO_IMGS = "./images/"
THRESHOLD_QUERY = 5.5


class OpenFile:
    def search_file(self):
        files = glob.glob(PATH_TO_IMGS+"*.pgm")
        return files
    
    def make_descriptors(self, input_img_path, input_img_test_path):
        img_gray = cv2.imread(input_img_path, cv2.COLOR_BGR2GRAY)
        img_tst = cv2.imread(input_img_test_path, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret, img_tst = cv2.threshold(img_tst, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        threshold = cv2.bitwise_not(threshold)

        # #MASCARA COM INVERSO DA input_img
        # outer = np.zeros(threshold.shape, dtype="uint8")
        # contour = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contour = imutils.grab_contours(contour)
        # contour = sorted(contour, key=cv2.contourArea, reverse = True)[0]
        # cv2.drawContours(outer, [contour], -1, 255, -1)


        x,y = threshold.shape
        img_tst = cv2.resize(img_tst,(y,x))
        # img_result = cv2.subtract(outer,img_tst)
        
        # img_result = cv2.bitwise_not(img_result)
        # ret, img_result = cv2.threshold(img_result, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        
        # eucli_sum = 0
        # for xi in range(0,x-1):
        #     for yi in range(0,y-1):
        #         eucli_sum = eucli_sum + np.square((int(img_result[xi][yi]) - int(img_tst[xi][yi])))
        # eucli = np.sqrt(eucli_sum)
        # print(" | DISTANCIA EUCLIDINA ENTRE AS IMAGENS: " + str(eucli))

        fd_q, hog_image_query = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=False)

        fd_db, hog_image_database = hog(img_tst, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=False)

        #DISTÂNCIA EUCLIDIANA
        eucli_fd = 0
        for xi in range(0,len(fd_q)):
            eucli_fd += np.square(fd_q[xi] - fd_db[xi])
        eucli_fd = np.sqrt(eucli_fd)

        #ERRO MÉDIO QUADRÁTICO
        mse_fd = np.sum((fd_q-fd_db)**2)
        mse_fd /= len(fd_q)

        #DISTÂNCIA COSSENO
        cos_fd = distance.cosine(fd_q,fd_db)



        # print("DISTANCIA EUCLIDIANA DOS FDS: " + str(eucli_fd))
        # if (eucli_fd <= THRESHOLD_QUERY):
            # display_img = np.hstack((img_gray,img_tst,outer,hog_image_query, hog_image_database))
            # similar.append(eucli_fd)
            # cv2.imshow("RESULTS", cv2.resize(display_img,(1200,200)))
            # cv2.imwrite("output_"+str(cont)+".jpg",display_img)
            # cont_sv+=1
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        return (eucli_fd,input_img_test_path,mse_fd,cos_fd)

        
    def find_related_imgs(self,query,imgv):
        similar_fd = []
        for img in imgv:
            eucli_fd,img_path,mse_fd,cos_fd = of.make_descriptors(query,img)
            similar_fd.append((eucli_fd,img_path,mse_fd,cos_fd))
        return similar_fd

    def show_related_img(self,query,similar_fd,N,save_result):
        img = []
        ximg,yimg = cv2.imread(query,cv2.COLOR_BGR2GRAY).shape
        for x in range(0,N):
            img.append(cv2.resize(cv2.imread(similar_fd[x][1],cv2.COLOR_BGR2GRAY),(yimg,ximg)))
            print(similar_fd[x])
        if save_result:
            # cv2.imwrite("./results/teste_bird.jpg",np.concatenate(img,axis=1))
            cv2.imshow("RESULTADOS",np.concatenate(img,axis=1))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    #CBIR(input_img,N) return N[maxS -> minS]
    #if input_img in I, maxS = input_img
    def CBIR(self,input_img,N,descriptors):
        
        print("TEST PURPOSE")

""""
=================ARGUMENTOSS PARA RODAR O PROGRAMA=================
    python3 script.py argv[1] argv[2]
    - argv[1] = INPUT_IMG (Somente o nome sem a extensão)
    - argv[2] = N (Quantidade de imagens a serem retornadas)
    - argv[3] = SAVE_RESULT (0 - Não salva a imagem com os correspondentes | 1 - Salva a imagem com os correpondentes)
"""


if __name__ == "__main__":
    of = OpenFile()
    img_files = of.search_file()
    query = PATH_TO_IMGS+str(sys.argv[1])+".pgm"
    N = sys.argv[2]

    # cont = 0
    # eucli_fd = 0
    similar_fd = []
    
    # for img in img_files:
    #     eucli_fd,img_path = of.make_descriptors(query,img,cont)
    #     # if (eucli_fd <= THRESHOLD_QUERY):
    #     similar_fd.append((eucli_fd,img_path))
    #     cont+=1

    similar_fd = of.find_related_imgs(query,img_files)

    similar_fd.sort(key = operator.itemgetter(0)) 
    save_result = sys.argv[3]
    of.show_related_img(query,similar_fd,int(N),int(save_result))

    similar_fd.sort(key = operator.itemgetter(2)) 
    of.show_related_img(query,similar_fd,int(N),int(save_result))
    
    similar_fd.sort(key = operator.itemgetter(3)) 
    of.show_related_img(query,similar_fd,int(N),int(save_result))
    
    # img = []
    # ximg,yimg = cv2.imread(query,cv2.COLOR_BGR2GRAY).shape
    # for x in range(0,N):
    #     img.append(cv2.resize(cv2.imread(similar_fd[x][1],cv2.COLOR_BGR2GRAY),(yimg,ximg)))
    # cv2.imshow("RESULTADOS",np.concatenate(img,axis=1))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("teste.jpg",np.concatenate(img,axis=1))
