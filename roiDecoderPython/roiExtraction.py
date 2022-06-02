from read_roi import read_roi_file
from read_roi import read_roi_zip

p = r"C:\Users\Windows\Desktop\1905-Practice\Madalyn-v1_1905_0019_dsRed.zip"
a = read_roi_zip(p)

z = r"C:\Users\Windows\Desktop\1905-Practice\Madalyn-v1_1905_0019_dsRed\Right+%28038%29\ACAd1\P1.roi"
zout = read_roi_file(z)
a= read_roi_file

shape = zout['P1']['paths']

alpha = np.zeros()

import tifffile as tiff
f = r"C:\Users\Windows\Desktop\1905-Practice\Madalyn-v1_1905_0022_dsRed.tif"
a = tiff.imread(f)


b = np.zeros([np.shape(a)[0], np.shape(a)[1]])







image_shape = (128, 128)
# image = np.random.random([128,128])
image = np.random.randint(0,255, size=[128,128])
polygon = np.array([[60, 100], [100, 40], [40, 40]])
mask = polygon2mask(image_shape, polygon)
mask.shape



def read_roi_zip(fname):
    import zipfile
    with zipfile.ZipFile(fname) as zf:
        return [read_roi_file(zf.open(n)) for n in zf.namelist()]



image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
fpath = f.split('.')[0]
xdimIm = np.shape(image)[0]
ydimIm = np.shape(image)[1]
custdpi = 300
figure = plt.figure(frameon=False, figsize=(xdimIm / custdpi, ydimIm / custdpi))
im0 = apply_brightness_contrast(cv2.imread(f, cv2.IMREAD_GRAYSCALE), 80, 50)
plt.imshow(im0, cmap='gray')
plt.show()
plt.axis('off')
figure.savefig(fpath + '.png', bbox_inches=Bbox([[0.0, 0.0], [xdimIm / custdpi, ydimIm / custdpi]]), pad_inches=0,
               dpi=custdpi)