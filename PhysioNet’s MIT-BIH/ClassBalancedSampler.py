import numpy as np
from torch.utils.data.sampler import Sampler
class ClassBalancedSampler(Sampler):
    def __init__(self, ClassLabel, shuffle):
        '''
        ClassLabel[n]: the class label of sample-n in a dataset, a list
        assume label starts from min(ClassLabel) to max(ClassLabel)
        '''
        self.shuffle=shuffle
        self.ClassLabel=np.array(ClassLabel)
        self.ClassLabel-=self.ClassLabel.min()
        self.class_count= self.ClassLabel.max()+1
        self.SampleCount=np.zeros(self.class_count, dtype=np.int64)
        self.rng=np.random.RandomState(0)
        self.ClassIndex=np.arange(0, self.class_count)
        self.SampleIndex=[]
        for n in range(0, self.class_count):
            self.SampleIndex.append(np.where(self.ClassLabel==n)[0])
            self.SampleCount[n]= len(self.SampleIndex[n])

    def run(self):
        if self.shuffle == True:
            self.rng.shuffle(self.ClassIndex)
            for n in range(0, self.class_count):
                self.rng.shuffle(self.SampleIndex[n])
        #
        max_count=self.SampleCount.max()
        IndexTable=np.zeros((self.class_count, max_count), dtype=np.int64)
        for n in range(0, self.class_count):
            SampleIndex_n=self.SampleIndex[n]
            Ln=len(SampleIndex_n)
            if Ln < max_count:
                SampleIndex_n = np.tile(SampleIndex_n, int(max_count/Ln)+1)
            IndexTable[n]= SampleIndex_n[0:max_count]
        #
        IndexTable=IndexTable[self.ClassIndex,:]
        return IndexTable

    def __iter__(self):
        IndexTable = self.run()
        IndexList = IndexTable.T.reshape(-1)
        return iter(IndexList)

    def __len__(self):
        return self.class_count*self.SampleCount.max()