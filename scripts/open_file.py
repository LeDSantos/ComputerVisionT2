import glob
from itertools import count
import cv2
import numpy as np
import imutils
import operator

from skimage.feature import hog
from skimage import exposure

PATH_TO_IMGS = "./images/"
THRESHOLD_QUERY = 5.5


class OpenFile:
    def search_file(self):
        files = glob.glob(PATH_TO_IMGS+"*.pgm")
        return files
    
    def make_descriptors(self, input_img_path, input_img_test_path, N):
        img_gray = cv2.imread(input_img_path, cv2.COLOR_BGR2GRAY)
        img_tst = cv2.imread(input_img_test_path, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret, img_tst = cv2.threshold(img_tst, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        threshold = cv2.bitwise_not(threshold)
        
        #SOBEL PARA PEGAR O CONTORNO DA input_img
        # grad_sobel_x = cv2.Sobel(threshold, cv2.CV_16S, 1, 0, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        # grad_sobel_y = cv2.Sobel(threshold, cv2.CV_16S, 0, 1, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        # abs_grad_x = cv2.convertScaleAbs(grad_sobel_x)
        # abs_grad_y = cv2.convertScaleAbs(grad_sobel_y)
        # grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        #MASCARA COM INVERSO DA input_img
        outer = np.zeros(threshold.shape, dtype="uint8")
        contour = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = imutils.grab_contours(contour)
        contour = sorted(contour, key=cv2.contourArea, reverse = True)[0]
        cv2.drawContours(outer, [contour], -1, 255, -1)

        # cv2.imshow("SOBEL", cv2.resize(grad,(600,400)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # cv2.imshow("OUTER",outer)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # x,y = grad.shape
        # img_tst = cv2.resize(img_tst,(y,x))
        # img_result = grad - img_tst

        # cv2.imshow("RESULT_SOBEL", np.ones(outer.shape,dtype="uint8"))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        x,y = outer.shape
        img_tst = cv2.resize(img_tst,(y,x))
        # img_result = outer - img_tst
        img_result = cv2.subtract(outer,img_tst)
        # print(outer[0][0])
        # print(img_tst[0][0])
        # print(img_result[0][0])
        
        img_result = cv2.bitwise_not(img_result)
        ret, img_result = cv2.threshold(img_result, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        
        eucli_sum = 0
        for xi in range(0,x-1):
            for yi in range(0,y-1):
                # if np.square((int(img_result[xi][yi]) - int(img_tst[xi][yi]))) <= 1:
                #     eucli_sum = eucli_sum + 0
                # else:
                #     eucli_sum = eucli_sum + np.square((int(img_result[xi][yi]) - int(img_tst[xi][yi])))
                eucli_sum = eucli_sum + np.square((int(img_result[xi][yi]) - int(img_tst[xi][yi])))
        eucli = np.sqrt(eucli_sum)
        print(str(N) + " | DISTANCIA EUCLIDINA ENTRE AS IMAGENS: " + str(eucli))

        fd_q, hog_image_query = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=False)

        fd_db, hog_image_database = hog(img_tst, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=False)

        # print("LEN FD_Q: " + str(len(fd_q)) + " | " + "LEN FD_DB: " + str(len(fd_db)))
        eucli_fd = 0
        for xi in range(0,len(fd_q)):
            eucli_fd += np.square(fd_q[xi] - fd_db[xi])
        eucli_fd = np.sqrt(eucli_fd)

        print("DISTANCIA EUCLIDIANA DOS FDS: " + str(eucli_fd))
        if (eucli_fd <= THRESHOLD_QUERY):
            display_img = np.hstack((img_gray,img_tst,outer,hog_image_query, hog_image_database))
            # similar.append(eucli_fd)
            # cv2.imshow("RESULTS", cv2.resize(display_img,(1200,200)))
            # cv2.imwrite("output_"+str(cont)+".jpg",display_img)
            # cont_sv+=1
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        return (eucli_fd,input_img_test_path)

        
    
    #CBIR(input_img,N) return N[maxS -> minS]
    #if input_img in I, maxS = input_img
    def CBIR(self,input_img,N,descriptors):
        
        print("TEST PURPOSE")


if __name__ == "__main__":
    of = OpenFile()
    img_files = of.search_file()
    query = PATH_TO_IMGS+"fork01.pgm"
    cont = 0
    eucli_fd = 0
    similar_fd = []
    N = 6
    for img in img_files:
        eucli_fd,img_path = of.make_descriptors(query,img,cont)
        # if (eucli_fd <= THRESHOLD_QUERY):
        similar_fd.append((eucli_fd,img_path))
        cont+=1
    similar_fd.sort(key = operator.itemgetter(0))
    img = []
    ximg,yimg = cv2.imread(query,cv2.COLOR_BGR2GRAY).shape
    for x in range(0,N):
        img.append(cv2.resize(cv2.imread(similar_fd[x][1],cv2.COLOR_BGR2GRAY),(yimg,ximg)))
    
    cv2.imshow("RESULTADOS",np.concatenate(img,axis=1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("teste.jpg",np.concatenate(img,axis=1))
