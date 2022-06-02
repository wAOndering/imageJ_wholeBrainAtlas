#########################
##### RESOURCES
#########################

# https://www.napari-hub.org/plugins/napari-skimage-regionprops
# https://pypi.org/project/read-roi/
# https://github.com/jayunruh/napari_jroitools

#########################
##### RESOURCES
#########################
import tifffile as tiff # need to have the proper codecs for LZW compression decoding
import skimage
from read_roi import read_roi_file
from read_roi import read_roi_zip

file = r"C:\Users\Windows\Desktop\Madalyn-v1_1905_0019_dsRed.tif"
im = tiff.imread(file)

image_shape = (im.shape[0], im.shape[1])
# image = np.random.random([128,128])
image = np.random.randint(0,255, size=[128,128])
polygon = np.array([[60, 1000], [1000, 400], [400, 3000]])
mask = skimage.draw.polygon2mask(image_shape, polygon)
mask.shape

# https://github.com/jayunruh/napari_jroitools/blob/master/importroi.py
### this method is good to import roi 


zipFile = r"Y:\Madalyn\1905-Practice\Madalyn-v1_1905_0019_dsRed.zip"
t = RoiDecoder()
rois = t.readzip(zipFile)



def readzip(self,path):
    #read a zip file of rois
    from zipfile import ZipFile
    rois=[]
    with ZipFile(path,'r') as zipobj:
        flist=zipobj.namelist()
        for fname in flist:
            print(fname)
            with zipobj.open(fname) as fp:
                read_roi_file(fp)
                #barr=np.fromfile(fp,dtype='b',count=-1)
                barr=fp.read()
                barr=np.array(bytearray(barr))
                rois.append(self.readroibytes(barr,fname))
    return rois




a = read_roi_zip(p)

 

z = r"Y:\Madalyn\1905-Practice\Madalyn-v1_1905_0019_dsRed\Right+%28038%29\ACAd1\P1.roi"
zout = read_roi_file(z)
zout[list(zout)[0]].get('paths')

a= read_roi_file

 



 

 



 

 

def read_roi_zip(fname=zipFile):

import zipfile

with zipfile.ZipFile(fname) as zf:
    print(zf)

    ls = [read_roi_file(zf.open(n)) for n in zf.namelist()]

 
