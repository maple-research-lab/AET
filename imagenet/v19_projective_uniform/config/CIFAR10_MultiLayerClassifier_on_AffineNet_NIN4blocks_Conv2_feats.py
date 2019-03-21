batch_size   = 128

config = {}
# set the parameters related to the training and testing set
data_train_opt = {} 
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = False
data_train_opt['epoch_size'] = None
data_train_opt['random_sized_crop'] = False
data_train_opt['dataset_name'] = 'cifar10'
data_train_opt['split'] = 'train'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = False
data_test_opt['epoch_size'] = None
data_test_opt['random_sized_crop'] = False
data_test_opt['dataset_name'] = 'cifar10'
data_test_opt['split'] = 'test'

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt
config['max_num_epochs'] = 200

networks = {}
feat_net_opt = {'num_classes': 6, 'num_stages': 4, 'use_avg_on_conv3': False}
feat_pretrained_file = './experiments/CIFAR10_AffineNet_NIN4blocks/model_net_epoch1500'
networks['feat_extractor'] = {'def_file': 'architectures/NetworkInNetwork.py', 'pretrained': feat_pretrained_file, 'opt': feat_net_opt,  'optim_params': None} 

cls_net_optim_params = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(100, 0.1),(150, 0.01),(200, 0.001)]}
cls_net_opt = {'num_classes':10, 'nChannels':192*8*8, 'cls_type':'MultLayer'}
networks['classifier'] = {'def_file': 'architectures/NonLinearClassifier.py', 'pretrained': None, 'opt': cls_net_opt, 'optim_params': cls_net_optim_params}
config['out_feat_keys'] = ['conv2']

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions
config['algorithm_type'] = 'FeatureClassificationModel'
config['best_metric'] = 'prec1'
