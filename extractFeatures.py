import cv2
import json
import torchfile


def extract(types):
	if types=='train':
		_train = 'ic-data/train_label.json'
	elif types=='test':
		_train = 'ic-data/test_label.json'
	else:
		_train = 'ic-data/val_label.json'
	_sift = []
	_hog = []
	with open(_train) as f:
		_images = json.load(f)
		for k,v in _images.items():
			im = cv2.imread(k)
			print k
			s = cv2.SIFT()
			keypoints,des = s.detectAndCompute(im,None)
			_sift.append(keypoints)
			hog = cv2.HOGDescriptor()
			keypoints = hog.compute(im)
			_hog.append(keypoints)

	torch.save('feats/sift/'+types+'_features_labels.t7',_sift)
	torch.save('feats/hog/'+types+'_features_labels.t7',_sift)
	

extract('train')
extract('test')
extract('val')
