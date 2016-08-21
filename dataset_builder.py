import glob
import lmdb
import cv2 as cv
from classify_config import ClassifyConfig 

class datasetBuilder:
    def openLMDB(self):
        self._config.createDbDir()
        imageDbFile = self._config.imageDbPath()
        labelDbFile = self._config.labelDbPath()
        self._imgDb = lmdb.Environment(imageDbFile, map_size = self._mapSize)
        self._labelDb = lmdb.Environment(labelDbFile, map_size = self._mapSize)
        self._imgRecord = self._imgDb.begin(write=True, buffers=True)
        self._labelRecord = self._labelDb.begin(write=True, buffers=True)

    def __init__(self, config):
        self._config = config
        self._mapSize = 1000000000
        self._iter = 1
        self._flushIter = 1000 

    def closeLMDB(self):
        self._imgRecord.commit()
        self._imgDb.close()
        self._labelRecord.commit()
        self._labelDb.close()

    def writeLMDB(self, label, image):
        key = "%010d" % self._iter
        if self._iter % self._flushIter == 0:
            self._imgRecord.commit()
            self._imgRecord = self._imgDb.begin(write=True, buffers=True)
            self._labelRecord.commit()
            self._labelRecord = self._labelDb.begin(write=True, buffers=True)
        self._imgRecord.put(key, image)
        self._labelRecord.put(key, label.encode())

    def build(self):
        self.openLMDB()

        for item in self._config.localConfig['data']:
            images = glob.glob(item['path'] + "/*." + item['format'])
            for image in images:
                imgData = cv.imread(image,1)
                self.writeLMDB(item['label'], imgData)

        self.closeLMDB()

if __name__ == '__main__': 
    config = ClassifyConfig("./dataset1.cfg")
    dataset = datasetBuilder(config)
    dataset.build()
    print config.imageDbPath()

