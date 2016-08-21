import os
import shutil
import json

class ClassifyConfig:
    def load_json(self, configPath):
        with open(configPath, 'rt') as jsonFile:
            val = jsonFile.read()
        return json.loads(val)

    def __init__(self, localConfigPath):
        self.localConfig = self.load_json(localConfigPath);
        self.globalConfig = self.load_json("./global.cfg");

    def createDir(self, directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

    def createDbDir(self):
        self.createDir(self.localConfig['network']['dataset'])

    def imageDbPath(self):
        return self.localConfig['network']['dataset'] + "/image.lmdb"

    def labelDbPath(self):
        return self.localConfig['network']['dataset'] + "/label.lmdb"

    def createModelDir(self):
        self.createDir(self.localConfig['network']['model']['path'])

    def trainModelPath(self):
        return self.localConfig['network']['model']['path'] + "/train.prototxt"

    def testModelPath(self):
        return self.localConfig['network']['model']['path'] + "/test.prototxt"

    def solverPath(self):
        return self.localConfig['network']['model']['path'] + "/solver.prototxt"

    def snapshotPath(self):
        return self.localConfig['network']['snapshot']

    def numClasses(self):
        classes = 0
        for item in self.localConfig['data']:
            classes = classes+1
        return classes

    def modelParam(self, param):
        return self.localConfig['network']['model'][param]

if __name__ == '__main__':
    config = ClassifyConfig("./dataset1.cfg")
    print config.localConfig
