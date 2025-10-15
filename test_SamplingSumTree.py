
from SamplingSumTree import SamplingSumTree

import random

from tqdm import tqdm

def test_basic():
    tree = SamplingSumTree(10)

    for i in range(1, 20):
        tree.add(i)
        print(f'tree realSize: {tree.realSize()}')
    tree.showTree()
    print(f'check Verify: {tree.checkVerify()}')

def test_write():
    tree = SamplingSumTree(10)

    for i in range(1, 20):
        tree.add(i)
    
    for i in range(10):
        tree.write(random.random(), i)
    tree.showTree()
    print(f'check Verify: {tree.checkVerify()}')

def test_writeErr():
    tree = SamplingSumTree(10)

    for i in range(1, 10):
        try:
            tree.write(10.0, i)
        except BaseException as e:
            print(e)
    
    tree.add(10.0)
    tree.showTree()
    tree.write(20.0, 0)
    tree.showTree()
    print(tree.checkVerify())

def test_randomSampling():
    tree = SamplingSumTree(10)

    for i in range(1, 11):
        tree.add(i)

    samplingIter = 100000
    sampleSize = 1000
    count = [0 for _ in range(11)]
    for _ in tqdm(range(samplingIter)):
        values, indices = tree.randomSampling(sampleSize)

        for val in values:
            # print(val)
            count[int(val)] += 1
    
    for idx, c in enumerate(count):
        print(f'{idx}, {c}')

def test_randomWeightedSampling():
    tree = SamplingSumTree(10)

    for i in range(1, 11):
        tree.add(i)

    samplingIter = 10000
    sampleSize = 1000
    count = [0 for _ in range(11)]
    for _ in tqdm(range(samplingIter)):
        values, indices = tree.randomWeightedSampling(sampleSize)

        for val in values:
            # print(val)
            count[int(val)] += 1
    
    for idx, c in enumerate(count):
        pred = idx / sum(range(11))
        prob = c / (samplingIter * sampleSize)
        err = abs(pred - prob)
        print(f'{idx}, {c}, {pred}, {prob}, {err}')

if __name__ == '__main__':
    # test_basic()
    # test_write()
    # test_writeErr()
    #print(test_randomSampling())
    test_randomWeightedSampling()