import numpy as np
import glob
import h5py

from keras.preprocessing.image import load_img, img_to_array

inpath = './input'
outpath = './output'

orgs = []
masks = []

print('original img')
files = glob.glob(inpath+'/org/*.jpg')
for imgfile in files:
    print(imgfile)
    img = load_img(imgfile, target_size=(256,256))
    imgarray = img_to_array(img)
    orgs.append(imgarray)

print('mask img')
files = glob.glob(inpath+'/mask/*.jpg')
for imgfile in files:
    print(imgfile)
    img = load_img(imgfile, target_size=(256,256))
    imgarray = img_to_array(img)
    masks.append(imgarray)

perm = np.random.permutation(len(orgs))
orgs = np.array(orgs)[perm]
masks = np.array(masks)[perm]
threshold = len(orgs)//10*9
imgs = orgs[:threshold]
gimgs = masks[:threshold]
vimgs = orgs[threshold:]
vgimgs = masks[threshold:]
print('shapes')
print('org imgs  : ', imgs.shape)
print('mask imgs : ', gimgs.shape)
print('test org  : ', vimgs.shape)
print('test tset : ', vgimgs.shape)

outh5 = h5py.File(outpath+'/datasetimages.hdf5', 'w')
outh5.create_dataset('train_data_raw', data=imgs)
outh5.create_dataset('train_data_gen', data=gimgs)
outh5.create_dataset('val_data_raw', data=vimgs)
outh5.create_dataset('val_data_gen', data=vgimgs)
outh5.flush()
outh5.close()