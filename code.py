#!/usr/bin/python
# necessary imports

import cv2
import numpy as np 
import subprocess
import time

# complete path to the application
path_to_notepad = 'C:\\Windows\\System32\\notepad.exe'
path_to_vlc = 'C:\\Program Files (x86)\\VideoLAN\\VLC\\vlc.exe'
path_to_wordpad = 'C:\\Program Files\\Windows NT\\Accessories\\wordpad.exe'
path_to_calc = 'C:\\windows\\system32\\calc.exe'
path_to_chrome = 'C:\\Users\\lisha\\AppData\\Local\\Google\\Chrome\\Application\\chrome.exe'
path_to_paint = 'C:\\windows\\system32\\mspaint.exe'
path_to_media = 'C:\\Program Files (x86)\\Windows Media Player\\wmplayer.exe'
path_to_opera = 'C:\\Program Files (x86)\\Opera\\opera.exe'
path_to_hearts = 'C:\\Program Files\\Microsoft Games\\Hearts\\Hearts.exe'

# table containing pixel tuples for letters 
letters = [['o',41,9,7,9,9,0,2,3900,1200],['n',40, 11, 6,11, 5, 5 ,3,4500,1750],['w',41,6,7,11,9,8,4,3200,2800],['c',28,8,2,5,8,0,1,2800,1200],['p',38,11,9,0,8,4,3,3900,1900],['v', 34,8, 7,2, 10,4,3,2200,700],['v', 34,8, 7,2, 10,4,3,2200,700],['b',45,9, 8, 8, 8, 5, 4,5600,1700],['m',43,11,11,7,5,4,4,3700,900],['z',36,6,9,5,8,6,1,4250,2250]]

# look up table connecting letter and path
dic = [['o',path_to_opera],['v',path_to_vlc],['n',path_to_notepad],['w',path_to_wordpad],['c',path_to_chrome],['p',path_to_paint],['b',path_to_calc],['m',path_to_media],['z',path_to_hearts]]

# function to look up the letter and open corresponding application 
def application(letter1):
    key = -1
    for i in range(len(dic)):
        if dic[i][0]==letter1 :
            subprocess.Popen([dic[i][1]])
            print dic[i][1]
    key = cv2.waitKey()
    print key 
    if key == 113 :
        cv2.destroyAllWindows()
    else :
        cv2.destroyAllWindows()
        imageproc()
  
# used for sorting  
def getKey(item):
    return item[1]

# function to match the trace to one for the available letters
def match(tuple1):
    list = []
    index = -1
    matched = 0
    pixels = tuple1[1]
    potential = []
    for i in range(len(letters)):
        tuple = letters[i]
        maxp = tuple[8]+tuple[9]
        minp = tuple[8]-tuple[9]
        max = tuple[1]+7
        min = tuple[1]-7
        # check for total pixels in original and resized image
        if min<= pixels <= max :
            if minp <= tuple1[8] <= maxp :
                list.append(tuple)
                index += 1
    print list
    # no match, then return 0 
    if index == -1 :
        return 0
    # only one match, check for difference. If in range return letter, else return 0
    elif index == 0 :
        abs = 1
        for j in range(2,7,1):
            absd = 0
            if (list[0][j] == 0):
                if (tuple1[j] == 0):
                    matched += 1 

            absd = list[0][j] - tuple1[j]
            if absd < 0 :
                absd = -absd
            if absd > 0 :
                abs = abs * absd
        if abs < 20 :
            print abs
            return list[0][0]
        return 0
    # more than one possibility, check for matched zero regions, unmatched zero regions and absolute difference 
    else :
        for i in range(0,len(list)):
            absdiff = 1
            notmatched = 0
            absd = 0
            tuple = list[i]
            matched = 0
            mixed = zip(tuple,tuple1)
            
            for j in range(2,7,1):
                absd = 0
                if (mixed[j][1] == 0):
                    if (mixed [j][0] == 0):
                        matched += 1
                    else :
                        notmatched +=1  
    
                absd = mixed[j][0]-mixed[j][1]
                if absd < 0 :
                    absd = -absd
                if absd > 0 :
                    absdiff = absdiff*absd
            potential.append([tuple[0],matched,absdiff,notmatched])
        # sort according to most zero regions matched and least absolute difference
        potential.sort(key = lambda row: (row[1],-row[2]), reverse = True) 
        print potential
        index = 1
        LENGTH = len(potential) 
        for index in range(LENGTH):
            #check for unmatched regions
            if potential[index][3] == 0 :
                if potential[index][2] <= 20 :
                    return potential[index][0]
        return 0
                                 
# function to manipulate the trace, extract region-wise pixel count and display results
def idletter(trace):
    img1 = cv2.cvtColor(trace, cv2.COLOR_BGR2GRAY) 
    cv2.imshow('original',img1)
    
    # total pixels in original trace
    tot = cv2.countNonZero(img1) 
    print tot
    
    #threshold and crop to get a bounding rectangle for the trace
    _,thresh = cv2.threshold(img1,1,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas) 
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img1[y:y+h,x:x+w]
    cv2.imshow('cropped',crop)
    
    # resize the image
    final = cv2.resize(crop,(9,14), interpolation = cv2.INTER_AREA)
    th2 = cv2.adaptiveThreshold(final,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,2)
    
    # calculate pixels in each region
    pixels = cv2.countNonZero(th2)
    topleft = th2[0:6,0:4]
    topleftp = cv2.countNonZero(topleft)
    topright = th2[0:6,5:9]
    toprightp = cv2.countNonZero(topright)
    bottomright = th2[8:15,5:9]
    bottomrightp = cv2.countNonZero(bottomright)
    bottomleft = th2[8:15,0:4]
    bottomleftp = cv2.countNonZero(bottomleft)
    middle = th2[5:9,3:6]
    middlep = cv2.countNonZero(middle)
    middleline = th2[6:7,0:9]
    middlelinep = cv2.countNonZero(middleline)
    t = ['x',pixels,topleftp,toprightp,bottomrightp,bottomleftp,middlep,middlelinep,tot,0]
    print t
    # check for total pixels
    if tot <20000 :
        letter = match(t)
    else :
        letter = 0

    if letter!=0 :
        print "letter is " + letter
        application(letter)
    else: 
        print "no matches"
        application('x')

    cv2.imwrite('final1.jpg',th2)
    cv2.waitKey(0)  


# function to subtract background and frames
def extract(imgbg,imgfg):
    # split the images into RGB channels 
    b1,g1,r1 = cv2.split(imgbg)
    b2,g2,r2 = cv2.split(imgfg)

    # find absolute difference between respective channels 
    bb = cv2.absdiff(b1,b2)
    gg = cv2.absdiff(g1,g2)
    rr = cv2.absdiff(r1,r2)

    # threshold each channel
    ret1, b = cv2.threshold(bb,50,255,cv2.THRESH_BINARY)
    ret2, g = cv2.threshold(gg,50,255,cv2.THRESH_BINARY) 
    ret3, r = cv2.threshold(rr,50,255,cv2.THRESH_BINARY)

    # merge and blur the image
    rgb = cv2.merge((r,g,b))
    cv2.medianBlur(rgb,3)
    return rgb

# function to set background
def setbg():
    camera = cv2.VideoCapture(0)
    retval, im = camera.read()
    avg1 = np.float32(im)
    print "setting background"
    for i in range(100): 
         retval, im = camera.read()
         cv2.accumulateWeighted(im,avg1,0.1)
         res1 = cv2.convertScaleAbs(avg1)
         cv2.waitKey(10)
    cv2.imshow("Background",res1)
    return res1
    del(camera)
    
# function to process video stream 
def imageproc():
        
        #set background
        imgbg = setbg()
        pattern = np.zeros(imgbg.shape,np.uint8)
        videoFrame1 = cv2.VideoCapture(0)
        keypressed = -1
        while(keypressed < 0):
            retu, img = videoFrame1.read()
            cv2.imshow("Camera",img)
            keypressed = cv2.waitKey(1)
        keypressed = -1
        count = 0
    
        while(keypressed < 0):
            count += 1
            readSucsess, imgfg = videoFrame1.read()
            cv2.imshow("Camera",imgfg)
            keypressed = cv2.waitKey(2)
            rgb = extract(imgbg,imgfg)
            #convert to grayscale
            gray= cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
            ret,thresh1 = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas) 
            #extract biggest contour and topmost point of that
            cnt=contours[max_index]
            topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
            #draw it onto a black background
            drawing = np.zeros(imgbg.shape,np.uint8)
            cv2.drawContours(drawing, contours, max_index, (0,255,0), 2)
            cv2.circle(drawing,topmost,1,(255,255,255),5)
             
            cv2.imshow("Hand Tracking",drawing)
               
            if(count == 1):
                cv2.circle(pattern,topmost,1,(255,255,255),5)
                prev = topmost
            else:  
                cv2.line(pattern,topmost,prev,(255,255,255),5)
                prev = topmost 
            cv2.imshow("Pattern",pattern)
        #pattern = cv2.flip(pattern,1)  
        cv2.imshow("Pattern",pattern)
        cv2.imwrite( "letter1.jpg", pattern )
        idletter(pattern)
        cv2.waitKey(0)
                
imageproc()
