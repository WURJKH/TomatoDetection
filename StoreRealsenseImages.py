import rospy
from datetime import datetime
import numpy as np
import cv2
import sensor_msgs.msg
import os
import time

class Ros:
    def __init__(self):
    
        self.writedir = "/home/agro/Documents/FinalScripts/Step1/RS_images"
        files = os.listdir(self.writedir)
        for f in files:
            os.remove(self.writedir+"/"+f)
            
            
        self.rgb = None
        self.depth = None
        self.starttime = time.time()
        
        rospy.Subscriber('/camera/color/image_raw', sensor_msgs.msg.Image, self.colorcallback, queue_size=1, buff_size=2**32)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', sensor_msgs.msg.Image, self.depthcallback, queue_size=1, buff_size=2**32)


       	

    def colorcallback(self, data):
        color_img = data
        shape = (color_img.height, color_img.width, 3) # 3 is the number of color channels
        img = np.frombuffer(color_img.data, dtype=np.uint8).reshape(shape)[...,::-1] # last part converts rgb format to bgr (for opencv2)
        self.rgb = img

        curtime = datetime.now()
        date = datetime.strptime(str(curtime), '%Y-%m-%d %H:%M:%S.%f')
        yyyy = str(date.year)
        mm = str(date.month).zfill(2)
        dd = str(date.day).zfill(2)
        hh = str(date.hour).zfill(2)
        MM = str(date.minute).zfill(2)
        ss = str(date.second).zfill(2)
        ms = str(date.microsecond)

        datetimestr = yyyy + mm + dd + "_" + hh + MM + ss + ms

        cv2.imwrite(os.path.join(self.writedir, datetimestr +'_rgb' +'.tiff'), img)
        
        if time.time() - 0.5 > self.starttime:
        	rospy.signal_shutdown("Klaar")


    def depthcallback(self, data):
        depth_img = data
        shape = (depth_img.height, depth_img.width)
        dimg = np.frombuffer(depth_img.data, dtype=np.uint16).reshape(shape)
        self.depth = dimg

        curtime = datetime.now()
        date = datetime.strptime(str(curtime), '%Y-%m-%d %H:%M:%S.%f')
        yyyy = str(date.year)
        mm = str(date.month).zfill(2)
        dd = str(date.day).zfill(2)
        hh = str(date.hour).zfill(2)
        MM = str(date.minute).zfill(2)
        ss = str(date.second).zfill(2)
        ms = str(date.microsecond)

        datetimestr = yyyy + mm + dd + "_" + hh + MM + ss + ms

        cv2.imwrite(os.path.join(self.writedir, datetimestr +'_depth' +'.tiff'), dimg)


if __name__ == "__main__":
    rospy.init_node('communication',anonymous=True)

    roscoms = Ros()
    rospy.spin()
    
