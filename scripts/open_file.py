import glob
from itertools import count
import cv2
import numpy as np
import imutils
import operator
import sys
from scipy.spatial import distance
import re
import math

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

        #DESCRITOR HOG
        fd_q, hog_image_query = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=False)

        fd_db, hog_image_database = hog(img_tst, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=False)

        #DISTÂNCIA EUCLIDIANA
        eucli_fd = self.euclidean_1D(fd_q,fd_db)
        # eucli_fd = self.euclidean_2D(hog_image_query,hog_image_database)
        # for xi in range(0,len(fd_q)):
        #     eucli_fd += np.square(fd_q[xi] - fd_db[xi])
        # eucli_fd = np.sqrt(eucli_fd)

        #ERRO MÉDIO QUADRÁTICO
        mse_fd = self.mse_1d(fd_q,fd_db)
        # mse_fd = np.sum((fd_q-fd_db)**2)
        # mse_fd /= len(fd_q)

        #DISTÂNCIA COSSENO
        cos_fd = distance.cosine(fd_q,fd_db)


        #GAUSSIAN BLUR PARA AUMENTO DA IMAGEM
        img_blur1_input = cv2.GaussianBlur(img_gray,(1,1),0)
        erro1_mse = self.mse_2imgs(img_tst,img_blur1_input,input_img_path,"IMG_BLUR1")
        blur1_eucli = self.euclidean_2D(img_blur1_input,img_tst)
        cos_blur1 = distance.cosine(img_blur1_input.flatten(),img_tst.flatten())

        img_blur3_input = cv2.GaussianBlur(img_gray,(3,3),0)
        erro3_mse = self.mse_2imgs(img_tst,img_blur3_input,input_img_path,"IMG_BLUR3")
        blur3_eucli = self.euclidean_2D(img_blur3_input,img_tst)
        cos_blur3 = distance.cosine(img_blur1_input.flatten(),img_tst.flatten())
        
        img_blur5_input = cv2.GaussianBlur(img_gray,(5,5),0)
        erro5_mse = self.mse_2imgs(img_tst,img_blur5_input,input_img_path,"IMG_BLUR5")
        blur5_eucli = self.euclidean_2D(img_blur5_input,img_tst)
        cos_blur5 = distance.cosine(img_blur1_input.flatten(),img_tst.flatten())

        # cv2.imshow("BLUR", cv2.resize(cv2.hconcat([img_blur1_input,img_blur3_input,img_blur5_input]),(600,400)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #ROTAÇÃO NA IMAGEM
        rotate_values = []
        for ind in range(45,360,45):
            rotaded = self.Random_RotationScale(img_gray,ind)
            img_tst_resize = cv2.resize(img_tst,(rotaded.shape[0],rotaded.shape[1]))
            if rotaded.shape[0] != img_tst_resize.shape[0]:
                img_tst_resize = cv2.resize(img_tst,(rotaded.shape[1],rotaded.shape[0]))
            # print("ROTADED SHAPE: " + str(rotaded.shape) + " IMG_TST SHAPE: " + str(img_tst_resize.shape))
            rotate_eucli = self.euclidean_2D(rotaded,img_tst_resize)
            rotate_values.append(rotate_eucli)
        
        rot45_eucli = rotate_values[0]
        rot90_eucli = rotate_values[1]
        rot135_eucli = rotate_values[2]
        rot180_eucli = rotate_values[3]
        rot225_eucli = rotate_values[4]
        rot270_eucli = rotate_values[5]
        rot315_eucli = rotate_values[6]
        

        #CANNY
        img_blur3_db = cv2.GaussianBlur(img_tst,(3,3),0)
        canny_input = cv2.Canny(img_blur3_input,225,250)
        canny_db = cv2.Canny(img_blur3_db,225,250)
        canny_eucli = self.euclidean_2D(canny_input,canny_db)
        # canny = cv2.hconcat([canny_db,canny_input])
        # cv2.imshow("CANNY", cv2.resize(canny,(600,400)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print("EUCLIDEAN CANNY: " + str(self.euclidean_2D(canny_input,canny_db)))
        # print("EUCLIDEAN CANNY FLATTENED: " + str(self.euclidean_1D(canny_input.flatten(),canny_db.flatten())))
        # print("EUCLIDEAN CANNY FLATTENED: " + str(distance.cityblock(canny_input.flatten(),canny_db.flatten())))
        
        # print("EUCLIDEAN CANNY INV: " + str(self.euclidean_2D(canny_input_inv,canny_db_inv)))
        # print("EUCLIDEAN CANNY INV FLATTENED: " + str(self.euclidean_1D(canny_input_inv.flatten(),canny_db_inv.flatten())))

        # print("DISTANCIA EUCLIDIANA DOS FDS: " + str(eucli_fd))
        # if (eucli_fd <= THRESHOLD_QUERY):
            # display_img = np.hstack((img_gray,img_tst,outer,hog_image_query, hog_image_database))
            # similar.append(eucli_fd)
            # cv2.imshow("RESULTS", cv2.resize(display_img,(1200,200)))
            # cv2.imwrite("output_"+str(cont)+".jpg",display_img)
            # cont_sv+=1
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        # return (eucli_fd,input_img_test_path,mse_fd,cos_fd)
        return (eucli_fd,input_img_test_path,mse_fd,cos_fd,blur1_eucli,blur3_eucli,blur5_eucli,rot45_eucli,rot90_eucli,rot135_eucli,
        rot180_eucli,rot225_eucli,rot270_eucli,rot315_eucli,canny_eucli,erro1_mse,erro3_mse,erro5_mse,
        cos_blur1,cos_blur3,cos_blur5)

    def mse_2imgs(self,imgA,imgB,query_name,prefix):
        erro = np.sum((imgA.astype("float") - imgB.astype("float")) **2)
        erro /= float(imgA.shape[0] * imgA.shape[1])
        # print(str(prefix) + " | IMAGEM ORIGINAL - " + str(query_name) + " | erro: " + str(erro))
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

    def find_related_imgs(self,query,imgv):
        similar_fd = []
        for img in imgv:
            # eucli_fd,img_path,mse_fd,cos_fd = of.make_descriptors(query,img)
            [eucli_fd,img_path,mse_fd,cos_fd,blur1_eucli,blur3_eucli,blur5_eucli,
            rot45_eucli,rot90_eucli,rot135_eucli,rot180_eucli,rot225_eucli,rot270_eucli,
            rot315_eucli,canny_eucli,erro1_mse,erro3_mse,erro5_mse,cos_blur1,cos_blur3,cos_blur5] = of.make_descriptors(query,img)
            
            # similar_fd.append((eucli_fd,img_path,mse_fd,cos_fd))
            similar_fd.append((img_path,eucli_fd,mse_fd,cos_fd,blur1_eucli,blur3_eucli,blur5_eucli,rot45_eucli,
            rot90_eucli,rot135_eucli,rot180_eucli,rot225_eucli,rot270_eucli,rot315_eucli,canny_eucli,erro1_mse,
            erro3_mse,erro5_mse,cos_blur1,cos_blur3,cos_blur5))
        return similar_fd

    def show_related_img(self,query,similar_fd,N,save_result):
        img = []
        open_img = cv2.imread(query,cv2.COLOR_BGR2GRAY)
        ximg,yimg = open_img.shape
        # print("(EUCLIDEAN_HOG,FILE,MSE_HOG,COS_HOG)")
        for x in range(0,N):
            img.append(cv2.resize(cv2.imread(similar_fd[x][0],cv2.COLOR_BGR2GRAY),(yimg,ximg)))
            # print(similar_fd[x])
        if save_result:
            # cv2.imwrite("./results/teste_bird.jpg",np.concatenate(img,axis=1))
            return np.concatenate(img,axis=1)
            cv2.imshow("RESULTADOS",np.concatenate(img,axis=1))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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

    def Random_RotationScale(self,original_img,rotation):
        scale = 1 # [0.85, 1.15]
        # rotation = random() * 360 # [0, 360 graus]
        # print("Scale: ",scale, " --- Rotation: ", rotation, "degrees")

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
            # of.show_related_img(query,similar_fd,int(N),int(save_result))
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
        else:
            similar_fd.sort(key = operator.itemgetter(np.argmax(P_VALUE)+1))
            img = self.show_related_img(PATH_TO_IMGS+str(sys.argv[1])+".pgm",similar_fd,int(N),1)
            cv2.imshow("RESULTADOS",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # best_P_VALUE = np.argmax(P_VALUE)+5
        # similar_fd.sort(key = operator.itemgetter(best_P_VALUE))
        return similar_fd
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

    similar_fd = of.evaluate_results(similar_fd,N,sys.argv[1])
    save_result = sys.argv[3]
    of.show_related_img(query,similar_fd,int(N),int(save_result))

    # similar_fd.sort(key = operator.itemgetter(1)) 
    # save_result = sys.argv[3]
    # of.show_related_img(query,similar_fd,int(N),int(save_result))
    # P,R,F = of.precision_recall(int(N),similar_fd,sys.argv[1])
    # print("EUCLIDEAN HOG SORT | P: " + str(P) + " | R: " + str(R) + " | F: " + str(F))

    # similar_fd.sort(key = operator.itemgetter(2)) 
    # of.show_related_img(query,similar_fd,int(N),int(save_result))
    # P,R,F = of.precision_recall(int(N),similar_fd,sys.argv[1])
    # print("MSE HOG SORT | P: " + str(P) + " | R: " + str(R) + " | F: " + str(F))
    
    # similar_fd.sort(key = operator.itemgetter(3)) 
    # of.show_related_img(query,similar_fd,int(N),int(save_result))
    # P,R,F = of.precision_recall(int(N),similar_fd,sys.argv[1])
    # print("COS HOG SORT | P: " + str(P) + " | R: " + str(R) + " | F: " + str(F))
    
    # img = []
    # ximg,yimg = cv2.imread(query,cv2.COLOR_BGR2GRAY).shape
    # for x in range(0,N):
    #     img.append(cv2.resize(cv2.imread(similar_fd[x][1],cv2.COLOR_BGR2GRAY),(yimg,ximg)))
    # cv2.imshow("RESULTADOS",np.concatenate(img,axis=1))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("teste.jpg",np.concatenate(img,axis=1))
