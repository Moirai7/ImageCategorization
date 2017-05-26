import torchfile
data = torchfile.load('feats/resnet/test_features_labels_resnet_finetune.t7')
print data.features, data.labels


