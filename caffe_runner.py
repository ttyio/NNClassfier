import caffe
from classify_config import ClassifyConfig 

class caffeRunner:
    def __init__(self, config):

    def doTrain(self):
        


if __name__ == '__main__': 
    config = ClassifyConfig("./dataset1.cfg")
    run = caffeRunner(config = config)
    run.doTrain()
