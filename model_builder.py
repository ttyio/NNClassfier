import caffe
from caffe import layers
from caffe import params
from caffe.proto import caffe_pb2
from classify_config import ClassifyConfig 

class modelBuilder:
    def __init__(self, config):
        self._config = config
        weight_param = dict(lr_mult=1, decay_mult=1)
        bias_param = dict(lr_mult=2,decay_mult=0)
        self.learned_param = [weight_param, bias_param]
        self.frozen_param = [dict(lr_mult=0)]*2

    def conv_relu(self, bottom, ks, nout, param, stride=1, pad=0, group=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0.1)):
        conv = layers.Convolution(bottom, kernel_size=ks, stride=stride,
                            num_output=nout, pad=pad, group=group,
                            param=param, weight_filler=weight_filler,
                            bias_filler=bias_filler)
        return conv, layers.ReLU(conv, in_place=True)
    
    def fc_relu(self, bottom, nout, param, weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=0.1)):
        fc = layers.InnerProduct(bottom, num_output=nout, param=param,
                                    weight_filler=weight_filler,
                                    bias_filler=bias_filler)
        return fc, layers.ReLU(fc, in_place=True)

    def max_pool(self, bottom, ks, stride=1):
        return layers.Pooling(bottom, pool=params.Pooling.MAX, kernel_size=ks, stride=stride)

    def caffenet(self, data, label=None, train=True, num_classes=1000,
                 classifier_name='fc8', learn_all=False):
        self.net = caffe.NetSpec()
        self.net.data = data
        param = self.learned_param if learn_all else self.frozen_param
        self.net.conv1, self.net.relu1 = self.conv_relu(self.net.data, 11, 96, stride=4, param=param)
        self.net.pool1 = self.max_pool(self.net.relu1, 3, stride=2)
        self.net.norm1 = layers.LRN(self.net.pool1, local_size=5, alpha=1e-4, beta=0.75)
        self.net.conv2, self.net.relu2 = self.conv_relu(self.net.norm1, 5, 256, pad=2, group=2, param=param)
        self.net.pool2 = self.max_pool(self.net.relu2, 3, stride=2)
        self.net.norm2 = layers.LRN(self.net.pool2, local_size=5, alpha=1e-4, beta=0.75)
        self.net.conv3, self.net.relu3 = self.conv_relu(self.net.norm2, 3, 384, pad=1, param=param)
        self.net.conv4, self.net.relu4 = self.conv_relu(self.net.relu3, 3, 384, pad=1, group=2, param=param)
        self.net.conv5, self.net.relu5 = self.conv_relu(self.net.relu4, 3, 256, pad=1, group=2, param=param)
        self.net.pool5 = self.max_pool(self.net.relu5, 3, stride=2)
        self.net.fc6, self.net.relu6 = self.fc_relu(self.net.pool5, 4096, param=param)
        if train:
            self.net.drop6 = fc7input = layers.Dropout(self.net.relu6, in_place=True)
        else:
            fc7input = self.net.relu6
        self.net.fc7, self.net.relu7 = self.fc_relu(fc7input, 4096, param=param)
        if train:
            self.net.drop7 = fc8input = layers.Dropout(self.net.relu7, in_place=True)
        else:
            fc8input = self.net.relu7
        fc8 = layers.InnerProduct(fc8input, num_output=num_classes, param=self.learned_param)
        self.net.__setattr__(classifier_name, fc8)
        if not train:
            self.net.probs = layers.Softmax(fc8)
        if label is not None:
            self.net.label = label
            self.net.loss = layers.SoftmaxWithLoss(fc8, self.net.label)
            self.net.acc = layers.Accuracy(fc8, self.net.label)
        return self.net.to_proto()

    def create_solver(self):
        s = caffe_pb2.SolverParameter()
        s.train_net = self._config.trainModelPath()
        #s.test_net.append(self._config.testModelPath())
        #s.test_interval = 1000  
        #s.test_iter.append(100) 
        s.iter_size = 1
        s.max_iter = 100000
        s.type = 'SGD'
        s.base_lr = float(self._config.modelParam('baseLr'))
        s.lr_policy = 'step'
        s.gamma = 0.1
        s.stepsize = 20000
        s.momentum = 0.9
        s.weight_decay = 5e-4
        s.display = 1000
        s.snapshot = 10000
        s.snapshot_prefix = self._config.snapshotPath()
        s.solver_mode = caffe_pb2.SolverParameter.GPU
        with open(self._config.solverPath(), 'w') as f:
            f.write(str(s))
        return f.name

    def create_net(self, learn_all=False, subset=None):
        train=True
        if subset is None:
            subset = 'train' if train else 'test'
        self._config.createModelDir()
        outputFile = self._config.trainModelPath()
        source = self._config.imageDbPath()
        numClasses = self._config.numClasses()
        # meanFile = '%CAFFE_MEAN_FILE%'
        # transform_param = dict(mirror=train, crop_size=227, mean_file=meanFile)
        transform_param = dict(mirror=train)
        style_data, style_label = layers.ImageData(transform_param=transform_param, source=source, batch_size=50, new_height=256, new_width=256, ntop=2)
        self.caffenet(data=style_data, label=style_label, train=train, num_classes=numClasses, classifier_name='fc8_custom', learn_all=learn_all)
        with open(outputFile, 'w') as f:
            f.write(str(self.net.to_proto()))
        self.create_solver()
        return 

if __name__ == '__main__': 
    config = ClassifyConfig("./dataset1.cfg")
    builder = modelBuilder(config = config);
    builder.create_net()
    print config.trainModelPath()
