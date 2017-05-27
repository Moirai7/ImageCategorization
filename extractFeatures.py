import cv2
import json
import torch
import numpy as np
from numpy.linalg import norm

def ComputeHog (img):
    # Do edge detection.
    height = len(img)
    width = len(img[0])
    features = []
    window_size = 8
    stride = 4
    for y in xrange(0, height - window_size, stride):
        for x in xrange(0, width - window_size, stride):
            wnd = img[y:y+window_size, x:x+window_size]
            # Do edge detection.
            gx = cv2.Sobel(wnd, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(wnd, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy)
            # Bin the angles.
            # TODO: we might want to try different rotations of the image.
            bin_n = 9
            bin = np.int32(bin_n*ang/(2*np.pi))
            # the magnitudes are used as weights for the gradient values.
            hist = np.bincount(bin.ravel(), mag.ravel(), bin_n)

            # transform to Hellinger kernel
            eps = 1e-7
            hist /= hist.sum() + eps
            hist = np.sqrt(hist)
            hist /= norm(hist) + eps
            features.extend(hist)
    return features

def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])     
        ++i
        temp_array.append(temp)
    return temp_array

def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)

def pad_to_dense(data1,data2,data3):
    M = np.concatenate((data1,data2,data3),axis=0)
    maxlen = max(len(r) for r in M)

    Z1 = np.zeros((len(data1), maxlen))
    for enu, row in enumerate(data1):
        Z1[enu, :len(row)] += row 
    Z2 = np.zeros((len(data2), maxlen))
    for enu, row in enumerate(data2):
        Z2[enu, :len(row)] += row
    Z3 = np.zeros((len(data3), maxlen))
    for enu, row in enumerate(data3):
        Z3[enu, :len(row)] += row
    return (Z1,Z2,Z3)

def extract(types):
	if types=='train':
		_train = 'ic-data/train_label.json'
	elif types=='test':
		_train = 'ic-data/test_label.json'
	else:
		_train = 'ic-data/val_label.json'
	_sift = []
	_hog = []
	_labels = []
	with open(_train) as f:
		_images = json.load(f)
		for k,v in _images.items():
			_labels.append(v)
			_path = 'ic-data/'+k.replace('\\','/')
			im = cv2.imread(_path)
			im = cv2.resize(im,(256,256))
			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  
			s = cv2.SIFT()
			keypoints,des = s.detectAndCompute(im,None)
			des = des.reshape(1, -1)[0].tolist()
			_sift.append(des)
			#hog = cv2.HOGDescriptor((32,64), (16,16), (8,8), (8,8), 9)
			#keypoints = hog.compute(im)
			features = ComputeHog (im)
			_hog.append(features)
	_labels = np.float32(_labels)
	np.save('feats/'+types+'_labels.npy',_labels)
	return (_sift,_hog)

def save(types,data1,data2,data3):
	data1,data2,data3 = pad_to_dense(data1,data2,data3)
	#_hog = pad_to_dense(_hog)
	np.save('feats/'+types+'/train_features_labels',data1)
	np.save('feats/'+types+'/test_features_labels',data2)
	np.save('feats/'+types+'/val_features_labels',data3)
	#np.save('feats/sift/'+types+'_features_labels',_sift)
	#torch.save(np.asarray(_sift),'feats/sift/'+types+'_features_labels.t7')
	#torch.save(np.asarray(_hog),'feats/hog/'+types+'_features_labels.t7')
	

_sift1,_hog1 = extract('train')
_sift2,_hog2 = extract('test')
_sift3,_hog3 = extract('val')

save('sift',_sift1,_sift2,_sift3)
save('hog',_hog1,_hog2,_hog3)
