import cv2
import numpy as np
import pandas as pd
import os


class Bildverarbeitung:
    def __init__(self, Img):
        try:
            self.img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
        except: 
            self.img = Img
        self.imgbin = np.zeros_like(self.img)
        self.imgregion = np.zeros_like(self.img)
        self.data= "Emty"
        self.metadata="Emty"
        self.mittekoor= np.zeros(2)
        self.solution_roa= 300
        self.regionofattraction_img =np.zeros((self.solution_roa,self.solution_roa))
        self.countur_img=cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        self.aussenkontur_img=np.zeros((self.solution_roa))
        self.edges= np.zeros_like(self.regionofattraction_img)
        self.blur = cv2.GaussianBlur(self.img,(5,5),0)

    def binar(self):   #Otsu mit Gaußfilterung         
        ret ,self.imgbin= cv2.threshold(cv2.bitwise_not(self.img),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #self.imgbin = cv2.adaptiveThreshold(self.img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        #self.imgbin = cv2.adaptiveThreshold(self.img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    def find_large_regions(self, threshold_area_small, threshold_area_big):
        contours, _ = cv2.findContours(self.imgbin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iteriere durch die Konturen und filtere Regionen basierend auf der Fläche
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter= cv2.arcLength(contour,True)
            if perimeter != 0:
                roundness= (4 * np.pi * area)/ (perimeter*perimeter)
                if area > threshold_area_small and area < threshold_area_big and roundness > 0.7:
                    # Zeichne die Region auf das Ergebnisbild
                    cv2.drawContours(self.imgregion, [contour], 0, 255, thickness=cv2.FILLED)
        
        kernel_size = 3
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        center = kernel_size // 2  
        kernel[:, center] = 1
        kernel[center, :] = 1
        self.imgregion = cv2.erode(self.imgregion, kernel, iterations=3) 
        self.imgregion = cv2.dilate(self.imgregion, kernel, iterations=3)
        

    def flaechen_mittelpunkt(self):
        # Berechne den gewichteten Durchschnitt der Koordinaten
        Mr= cv2.moments(self.imgregion)
 
        # calculate x,y coordinate of center
        try:
            self.mittekoor[0]= int(Mr["m10"] / Mr["m00"])
            self.mittekoor[1] = int(Mr["m01"] / Mr["m00"])
            #print("x= " + str(self.mittekoor[0]) + " und y= " + str(self.mittekoor[1]))
            size=10
            thickness=3
            cv2.line(self.countur_img, (int(self.mittekoor[0] - size/2), int(self.mittekoor[1])), (int(self.mittekoor[0] + size/2), int(self.mittekoor[1])), (0, 255, 0), thickness)
            # Zeichne vertikale Linie
            cv2.line(self.countur_img, (int(self.mittekoor[0]), int(self.mittekoor[1] - size/2)), (int(self.mittekoor[0]), int(self.mittekoor[1] + size/2)), (0, 255, 0), thickness)
        except:
            print('Mitte nicht gefunden')

    def regionofattraction(self):
        self.binar()
        self.find_large_regions(300, 100000)
        self.flaechen_mittelpunkt()
        if all(self.mittekoor) != None:
            shift= (self.solution_roa)
            shift2= int( self.solution_roa/2)
            # Berechne die Koordinaten des Ausschnitts
            y = int(self.mittekoor[0] - shift)
            x = int(self.mittekoor[1] - shift)
            try:
                if y <= 0 and x > 0:
                    self.regionofattraction_img= self.img[0:shift, abs(x):abs(x)+shift2]
                elif y > 0 and x <= 0:
                    self.regionofattraction_img = self.img[abs(y):abs(y)+shift2, 0:shift]
                elif y <= 0 and x <= 0:
                    self.regionofattraction_img = self.img[0:shift, 0:shift]
                else:
                    self.regionofattraction_img = self.img[abs(y):abs(y)+shift2, abs(x):abs(x)+shift2] 
            except:
                print('error')

    def aussenkontur(self):

        kernel_size = 3
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        center = kernel_size // 2  
        kernel[:, center] = 1
        kernel[center, :] = 1

        img_erosion = cv2.erode(self.imgregion, kernel, iterations=1) 
        img_dilation = cv2.dilate(self.imgregion, kernel, iterations=1) 
        self.aussenkontur_img = cv2.subtract(img_dilation , img_erosion )      
        
    def sobel_img(self):
    # Sobel Edge Detection
        img = cv2.adaptiveThreshold(self.img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        sobelx = cv2.Sobel(src=cv2.GaussianBlur(img,(5,5),0), ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3) 
        sobely = cv2.Sobel(src=cv2.GaussianBlur(img,(5,5),0), ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3) 

        sobel = cv2.Sobel(src=cv2.GaussianBlur(img,(5,5),0), ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        self.edges = sobel
        #sobel = cv2.magnitude(sobelx, sobely)
        #self.edges = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
        #sobel= np.unit8(sobel)
        #self.edges = cv2.Threshold(sobel, 50, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #contours, _ = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        threshold_area_small=10
        threshold_area_big=10000000
        # Iteriere durch die Konturen und filtere Regionen basierend auf der Fläche
        #for contour in contours:
            #area = cv2.contourArea(contour)
            #perimeter= cv2.arcLength(contour,True)
            
            #if area > threshold_area_small and area < threshold_area_big:
                #cv2.drawContours(self.edges, [contour], 0, 255, thickness=cv2.FILLED)      




    def Hough_Circles(self):
        # Use the Hough Circle Transform
        circles = cv2.HoughCircles(
            cv2.adaptiveThreshold(self.img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2),#self.blur,
            cv2.HOUGH_GRADIENT,
            dp=1,           # Inverse ratio of the accumulator resolution to the image resolution
            minDist=50,     # Minimum distance between detected centers
            param1=200,     # Upper threshold for the internal Canny edge detector
            param2=5,      # Threshold for center detection
            minRadius=5,   # Minimum radius to be detected
            maxRadius=100    # Maximum radius to be detected
        )

        # If circles are found, draw them on the original image
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw the outer circle
                cv2.circle(self.countur_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(self.countur_img, (i[0], i[1]), 2, (0, 0, 255), 3)
        
    def speichern(self, speicher_path, name, idx):
        cv2.imwrite('bilder/original'+ '_' + str(name) + str(idx) + '_' +'.png',self.img)
        cv2.imwrite('bilder/bin'+ '_' + str(name) + str(idx) + '_' +'.png',self.imgbin) 
        cv2.imwrite('bilder/region'+ '_' + str(name) + str(idx) + '_' +'.png',self.imgregion)
        cv2.imwrite('bilder/regionattraction'+ '_' + str(name) + str(idx) + '_' +'.png',self.regionofattraction_img )
        cv2.imwrite('bilder/countur'+ '_' + str(name) + str(idx) + '_' +'.png',self.countur_img)
        cv2.imwrite('bilder/aussenkontur'+ '_' + str(name) + str(idx) + '_' +'.png',self.aussenkontur_img)
        cv2.imwrite('bilder/sobel'+ '_' + str(name) + str(idx) + '_' +'.png',self.edges)
        return 0
        #self.data
        #self.metadata
        ##self.mittekoor
        


def main():
    imgr = cv2.imread("/bilder/original_right1_.png")
    imgl = cv2.imread("links.jpg")
    img= Bildverarbeitung(imgr)

    img.regionofattraction()
    img.aussenkontur()
    img.Hough_Circles()
    #img.mitte()

    
    cv2.imshow("test_image", img.imgregion)
    cv2.waitKey(0)
    #cv2.imshow("test_image", img.aussenkontur_img)
    #cv2.waitKey(0)
    cv2.imshow("test_image", img.countur_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    
    main()
    
    
    

    