
from typing import Tuple
import random
from multiprocessing import Pool

import numpy as np
from numpy.typing import NDArray

def _ascend_chunk(keys: NDArray[np.float64],
                  tree: NDArray[np.float64],
                  capacity: int) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """SumTree（配列長=2*capacity-1, 葉は [capacity-1 : 2*capacity-1)）を使って
    各 key に対する (sample, index) を返す。"""
    n = len(keys)
    samples = np.empty(n, dtype=np.float64)
    indices = np.empty(n, dtype=np.int64)

    for i, key in enumerate(keys):
        nidx = 0
        # ルートから葉へ
        while nidx < capacity - 1:
            lidx = 2 * nidx + 1
            ridx = lidx + 1
            left_sum = tree[lidx]
            if key < left_sum:
                nidx = lidx
            else:
                nidx = ridx
                key -= left_sum  # ← 右に行くときは左の和を引くのが正しい
        samples[i] = tree[nidx]
        indices[i] = nidx
    return samples, indices

class SamplingSumTree:

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._tree = np.empty(capacity * 2 - 1, dtype=np.float64)
        self._realSize = 0
        self._written = np.full(self._capacity, False, dtype=np.bool_)
        self._writeIndex = 0
        self._maxIndex = 0

    '''
    木への書き込み
    ただし，addによって既に書き込まれた範囲のみ
    '''
    def write(self, value: float, index: int) -> None:
        treeIndex = self._convLeafIndexToTreeIndex(index)

        if not self._written[index]:
            raise BaseException("SamplingSumTree.writeが未書き込み領域に書き込もうとしました")
        self._tree[treeIndex] = value

        self._propagateUpdate(treeIndex)

    '''
    複数値の書き込み
    Args:
        values: 書き込む値
        indices: 書き込む位置
    '''
    def multiWrite(self,  
                   values: NDArray[np.float64],
                   indices: NDArray[np.int16]) -> None:
        for value, index in zip(values, indices):
            self.write(value, index)

    '''
    その位置が未書き込み領域でないか(書き込み可能か)を判定

    Args:
        index: 確認する位置
    
    Return:
        書き込み可能であればTrue
    '''
    def canWrite(self, index: int) -> bool:
        return self._written[index]

    '''
    木の葉に先頭から順番に書き込む

    Args:
        value: 書き込む値
    
    Return:
        書き込んだ位置
    '''
    def add(self, value: float) -> int:
        self._written[self._writeIndex] = True
        self.write(value, self._writeIndex)

        retIndex = self._writeIndex
        self._addStep()

        return retIndex

    '''
    複数値の書き込み

    Args:
        values: 書き込む値
    
    Return:
        NDArray[np.int16]: 書き込んだ位置
    '''
    def multiAdd(self, values: NDArray[np.float64]) -> NDArray[np.int16]:
        retIndices = np.empty(len(values), dtype=np.int16)
        for idx, value in enumerate(values):
            retIndices[idx] = self.add(value)
        return retIndices

    '''
    読み取り

    Args: 
        読み取る位置
    '''
    def read(self, index: int) -> NDArray[np.float64]:
        treeIndex = self._convLeafIndexToTreeIndex(index)

        return self._tree[treeIndex]
    
    '''
    複数個所の読み取り

    Args:
        読み取る位置
    '''
    def multiRead(self, indices: NDArray[np.int16]) -> NDArray[np.float64]:
        vectorRead = np.frompyfunc(self.read, 1, 1)
        return vectorRead(indices)

    '''
    一様ランダムサンプリング
    '''
    def randomSampling(self, sampleSize: int) -> Tuple[NDArray[np.float64], NDArray[np.int16]]:
        sampleIndices = np.random.randint(0, self._realSize, sampleSize, dtype=np.int16)
        return (self.multiRead(sampleIndices), sampleIndices)
        
    '''
    重み付きサンプリング．重みは葉の値

    Arg:
        samplesize: サンプルサイズ
    
    Return:
        NDarray[np.float64]: サンプルした値
        NDArray[np.int16]: サンプルの位置
    '''
    def randomWeightedSampling(self, sampleSize: int) -> Tuple[NDArray[np.float64], NDArray[np.int16]]:
        keys = np.random.random(sampleSize) * self.total()

        samples = np.empty(sampleSize, dtype=np.float64)
        indices = np.empty(sampleSize, dtype=np.int16)
        for idx, key in enumerate(keys):
            sample, index = self._ascend(key)
            samples[idx] = sample
            indices[idx] = index

        return (samples, indices)

    def total(self) -> np.float64:
        return self._tree[0]

    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._realSize

    def realSize(self) -> int:
        return len(self)

    ########## in class utility

    ##### convert index

    def _parentIndex(self, childIndex: int) -> int:
        return int((childIndex - 1) / 2)

    def _lchildIndex(self, parentIndex: int) -> int:
        return 2 * parentIndex + 1

    def _rchildIndex(self, parentIndex: int) -> int:
        return 2 * parentIndex + 2

    def _convLeafIndexToTreeIndex(self, leafIndex: int) -> int:
        return leafIndex + self._capacity - 1

    def _convTreeIndexToLeafIndex(self, treeIndex: int) -> int:
        return treeIndex - self._capacity + 1

    ##### get parent or child
    def _getParentValue(self, childIndex: int) -> np.float64:
        return self._tree[self._parentIndex(childIndex)]

    def _getLchildValue(self, parentIndex: int) -> np.float64:
        return self._tree[self._lchildIndex(parentIndex)]

    def _getRchildValue(self, parentIndex: int) -> np.float64:
        return self._tree[self._rchildIndex(parentIndex)]

    ##### propagate update

    '''
    木の再計算．treeIndexから単点更新を行う

    Args:
        treeIndex: 再計算を開始する位置
    '''
    def _propagateUpdate(self, treeIndex: int) -> None:
        parentIndex = self._parentIndex(treeIndex)
        pVal = self._getParentValue(treeIndex)
        lchildVal = self._getLchildValue(parentIndex)
        rchildVal = self._getRchildValue(parentIndex)
        diff = (lchildVal + rchildVal) - pVal

        cntIndex = treeIndex

        while cntIndex != 0:
            cntIndex = self._parentIndex(cntIndex)
            self._tree[cntIndex] += diff

    ##### ascend

    '''
    keyを使って木を登る
    '''
    def _ascend(self, key: float) -> Tuple[np.float64, np.int16]:
        cntIndex = 0
        key_npfloat64 = np.float64(key)

        while cntIndex < self._capacity - 1:
            lchildVal = self._getLchildValue(cntIndex)

            if key_npfloat64 <= lchildVal:
                cntIndex = self._lchildIndex(cntIndex)
            else:
                key_npfloat64 -= lchildVal
                cntIndex = self._rchildIndex(cntIndex)
        
        return (self._tree[cntIndex], np.int16(cntIndex))

    ##### step counter

    '''
    _writeIndexを進める
    '''
    def _stepWriteIndex(self):
        self._writeIndex = (self._writeIndex + 1) % self._capacity

    '''
    _realSizeを進める
    '''
    def _stepRealSize(self):
        self._realSize = min(self._realSize + 1, self._capacity)

    def _addStep(self):
        self._stepWriteIndex()
        self._stepRealSize()
    
    ########## debug

    def showTree(self):
        contentStr = ""
        for treeVal in self._tree:
            contentStr += f'{treeVal},'
        contentStr = contentStr[0:-2]
        print(contentStr)
    
    def checkVerify(self, errThreshold: np.float64=np.float64(1e-3)) -> int:
        for idx in range(self._capacity - 1):
            # print(idx)
            pVal = self._tree[idx]
            lchildVal = self._getLchildValue(idx)
            rchildVal = self._getRchildValue(idx)
            childSum = lchildVal + rchildVal

            # print(f'{idx}, {pVal}, {lchildVal}, {rchildVal}')
            if not(pVal - errThreshold <= childSum and childSum <= pVal + errThreshold):
                print("errer")
                return idx
        
        return -1
