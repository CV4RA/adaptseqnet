import torch
import torch.utils.data as data
import itertools
from os.path import join
from scipy.io import loadmat
import numpy as np
from collections import namedtuple
from sklearn.neighbors import NearestNeighbors
import faiss
import h5py

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
                                   'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
                                   'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr',
                                   'dbTimeStamp', 'qTimeStamp', 'gpsDb', 'gpsQ'])


class Dataset():
    def __init__(self, dataset_name, train_mat_file, test_mat_file, val_mat_file, opt):
        self.dataset_name = dataset_name
        self.train_mat_file = train_mat_file
        self.test_mat_file = test_mat_file
        self.val_mat_file = val_mat_file
        self.struct_dir = "./structFiles/"
        self.seqL = opt.seqL
        self.seqL_filterData = opt.seqL_filterData

        self.dbDescs = None
        self.qDescs = None
        self.trainInds = None
        self.valInds = None
        self.testInds = None
        self.db_seqBounds = None
        self.q_seqBounds = None

    def loadPreComputedDescriptors(self, ft1, ft2, seqBounds=None):
        self.dbDescs = ft1
        self.qDescs = ft2
        if seqBounds is None:
            self.db_seqBounds = None
            self.q_seqBounds = None
        else:
            self.db_seqBounds = seqBounds[0]
            self.q_seqBounds = seqBounds[1]
        return self.dbDescs.shape[1]

    def get_whole_training_set(self, onlyDB=False):
        structFile = join(self.struct_dir, self.train_mat_file)
        indsSplit = self.trainInds
        return WholeDatasetFromStruct(structFile, indsSplit, self.dbDescs, self.qDescs,
                                      seqL=self.seqL, onlyDB=onlyDB,
                                      seqBounds=[self.db_seqBounds, self.q_seqBounds],
                                      seqL_filterData=self.seqL_filterData)

    def get_whole_val_set(self):
        structFile = join(self.struct_dir, self.val_mat_file)
        indsSplit = self.valInds
        return WholeDatasetFromStruct(structFile, indsSplit, self.dbDescs, self.qDescs,
                                      seqL=self.seqL,
                                      seqBounds=[self.db_seqBounds, self.q_seqBounds],
                                      seqL_filterData=self.seqL_filterData)

    def get_whole_test_set(self):
        structFile = join(self.struct_dir, self.test_mat_file)
        indsSplit = self.testInds
        return WholeDatasetFromStruct(structFile, indsSplit, self.dbDescs, self.qDescs,
                                      seqL=self.seqL,
                                      seqBounds=[self.db_seqBounds, self.q_seqBounds],
                                      seqL_filterData=self.seqL_filterData)

    def get_training_query_set(self, margin=0.1, nNegSample=1000, use_regions=False):
        structFile = join(self.struct_dir, self.train_mat_file)
        indsSplit = self.trainInds
        return QueryDatasetFromStruct(structFile, indsSplit, self.dbDescs, self.qDescs,
                                      nNegSample=nNegSample,
                                      margin=margin,
                                      use_regions=use_regions,
                                      seqL=self.seqL,
                                      seqBounds=[self.db_seqBounds, self.q_seqBounds])

    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None, None, None, None, None, None, None

        query, positive, negatives1, negatives2, indices = zip(*batch)
        query = data.dataloader.default_collate(query)
        positive = data.dataloader.default_collate(positive)

        negCounts1 = torch.tensor([x.shape[0] for x in negatives1])
        negCounts2 = torch.tensor([x.shape[0] for x in negatives2])

        negatives1 = torch.cat(negatives1, 0)
        negatives2 = torch.cat(negatives2, 0)
        indices = list(itertools.chain(*indices))

        return query, positive, negatives1, negatives2, negCounts1, negCounts2, indices


class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, indsSplit, dbDescs, qDescs,
                 onlyDB=False, seqL=1, seqBounds=None, seqL_filterData=None):
        super().__init__()
        self.seqL = seqL
        self.filterBoundaryInds = False if seqL_filterData is None else True

        self.dbStruct = parse_db_struct(structFile)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.images = dbDescs[indsSplit[0]]

        if not onlyDB:
            qImages = qDescs[indsSplit[1]]
            self.images = np.concatenate([self.images, qImages], axis=0)

        if seqBounds[0] is None:
            db_seqBounds = np.array([[0, len(self.images[:len(indsSplit[0])])] for _ in range(len(indsSplit[0]))])
            q_seqBounds  = np.array([[len(db_seqBounds), len(db_seqBounds)+len(indsSplit[1])] for _ in range(len(indsSplit[1]))])
            self.seqBounds = np.vstack([db_seqBounds, q_seqBounds])
        else:
            db_seqBounds = seqBounds[0][indsSplit[0]]
            q_seqBounds  = seqBounds[1][indsSplit[1]] + db_seqBounds[-1,-1]
            self.seqBounds = np.vstack([db_seqBounds, q_seqBounds])

        self.validInds = np.arange(len(self.images))
        if self.filterBoundaryInds:
            validFlags = getValidSeqInds(self.seqBounds, seqL_filterData)
            self.validInds = np.argwhere(validFlags).flatten()

        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        index = self.validInds[index]
        sIdMin, sIdMax = self.seqBounds[index]
        img = self.images[getSeqInds(index, self.seqL, sIdMax, minNum=sIdMin)]
        return img, index

    def __len__(self):
        return len(self.validInds)

    def get_positives(self, retDists=False):
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)
            print("Using Localization Radius: ", self.dbStruct.posDistThr)
            self.distances, self.positives = knn.radius_neighbors(
                self.dbStruct.utmQ, radius=self.dbStruct.posDistThr
            )
        if retDists:
            return self.positives, self.distances
        else:
            return self.positives


class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, indsSplit, dbDescs, qDescs,
                 nNegSample=1000, nNeg=10, margin=0.1, use_regions=False,
                 seqL=1, seqBounds=None):
        super().__init__()
        self.seqL = seqL
        self.dbDescs = dbDescs[indsSplit[0]]
        self.qDescs = qDescs[indsSplit[1]]
        self.margin = margin
        self.nNegSample = nNegSample
        self.nNeg = nNeg
        self.use_faiss = True

        self.dbStruct = parse_db_struct(structFile)

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        if self.dbStruct.utmDb is None or self.dbStruct.utmQ is None:
            raise ValueError(f"Your structFile [{structFile}] is missing utmDb / utmQ coordinates. "
                             f"Please regenerate the .mat file with proper coordinate fields.")

        if seqBounds[0] is None:
            self.db_seqBounds = np.array([[0, len(self.dbDescs)] for _ in range(len(self.dbDescs))])
            self.q_seqBounds = np.array([[0, len(self.qDescs)] for _ in range(len(self.qDescs))])
        else:
            self.db_seqBounds = seqBounds[0][indsSplit[0]]
            self.q_seqBounds = seqBounds[1][indsSplit[1]]

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)
        self.nontrivial_distances, self.nontrivial_positives = knn.radius_neighbors(
            self.dbStruct.utmQ,
            radius=self.dbStruct.nonTrivPosDistSqThr ** 0.5,
            return_distance=True
        )
        self.nontrivial_positives = [np.sort(x) for x in self.nontrivial_positives]
        potential_pos = knn.radius_neighbors(self.dbStruct.utmQ,
                                             radius=self.dbStruct.posDistThr,
                                             return_distance=False)
        self.potential_negatives = [
            np.setdiff1d(np.arange(self.dbStruct.numDb), pos, assume_unique=True)
            for pos in potential_pos
        ]
        self.cache = None
        self.negCache = [np.empty((0,), dtype=np.int32) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, index):
        with h5py.File(self.cache, mode="r") as h5:
            h5feat = h5.get("features")
            qOffset = self.dbStruct.numDb
            qFeat = h5feat[index + qOffset]
            posFeat = h5feat[self.nontrivial_positives[index].tolist()]

            faiss_index = faiss.IndexFlatL2(posFeat.shape[1])
            faiss_index.add(posFeat)
            dPos, posNN = faiss_index.search(qFeat.reshape(1, -1), 1)
            dPos = np.sqrt(dPos)[0][0]
            posIndex = self.nontrivial_positives[index][posNN[0][0]]

            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample])).astype(np.int32)
            negSample = np.sort(negSample)

            negFeat = h5feat[negSample.tolist()]
            faiss_index = faiss.IndexFlatL2(negFeat.shape[1])
            faiss_index.add(negFeat)
            dNeg, negNN = faiss_index.search(qFeat.reshape(1, -1), self.nNeg * 10)
            dNeg = np.sqrt(dNeg).reshape(-1)
            negNN = negNN.reshape(-1)
            violatingNeg = dNeg < dPos + self.margin ** 0.5

            if np.sum(violatingNeg) < 2:
                return None

            negNN = negNN[violatingNeg]
            half = self.nNeg // 2
            negIndices1 = negSample[negNN[:half]].astype(np.int32)
            negIndices2 = negSample[negNN[half:self.nNeg]].astype(np.int32)
            if len(negIndices2) == 0:
                negIndices2 = negIndices1.copy()

            self.negCache[index] = np.concatenate([negIndices1, negIndices2])

        sIdMin_q, sIdMax_q = self.q_seqBounds[index]
        query = self.qDescs[getSeqInds(index, self.seqL, sIdMax_q, sIdMin_q)]
        sIdMin_p, sIdMax_p = self.db_seqBounds[posIndex]
        positive = self.dbDescs[getSeqInds(posIndex, self.seqL, sIdMax_p, sIdMin_p)]

        negatives1 = []
        for negIndex in negIndices1:
            sIdMin_n, sIdMax_n = self.db_seqBounds[negIndex]
            neg = torch.tensor(self.dbDescs[getSeqInds(negIndex, self.seqL, sIdMax_n, sIdMin_n)])
            negatives1.append(neg)
        negatives1 = torch.stack(negatives1, 0)

        negatives2 = []
        for negIndex in negIndices2:
            sIdMin_n, sIdMax_n = self.db_seqBounds[negIndex]
            neg = torch.tensor(self.dbDescs[getSeqInds(negIndex, self.seqL, sIdMax_n, sIdMin_n)])
            negatives2.append(neg)
        negatives2 = torch.stack(negatives2, 0)

        return query, positive, negatives1, negatives2, [index, posIndex] + negIndices1.tolist() + negIndices2.tolist()

    def __len__(self):
        return len(self.qDescs)


def getSeqInds(idx, seqL, maxNum, minNum=0, retLenDiff=False):
    seqLOrig = seqL
    seqInds = np.arange(max(minNum, idx - seqL // 2), min(idx + seqL - seqL // 2, maxNum), 1)
    lenDiff = seqLOrig - len(seqInds)
    if retLenDiff:
        return lenDiff
    if seqInds[0] == minNum:
        seqInds = np.concatenate([seqInds, np.arange(seqInds[-1]+1, seqInds[-1]+1+lenDiff, 1)])
    elif lenDiff > 0 and seqInds[-1] in range(maxNum-1,maxNum):
        seqInds = np.concatenate([np.arange(seqInds[0]-lenDiff, seqInds[0], 1), seqInds])
    return seqInds


def parse_db_struct(path):
    mat = loadmat(path)
    fieldnames = list(mat['dbStruct'][0, 0].dtype.names)
    dataset = mat['dbStruct'][0, 0]['dataset'].item()
    whichSet = mat['dbStruct'][0, 0]['whichSet'].item()
    dbImage = [f[0].item() for f in mat['dbStruct'][0, 0]['dbImageFns']]
    qImage = [f[0].item() for f in mat['dbStruct'][0, 0]['qImageFns']]
    numDb = mat['dbStruct'][0, 0]['numImages'].item()
    numQ = mat['dbStruct'][0, 0]['numQueries'].item()
    posDistThr = mat['dbStruct'][0, 0]['posDistThr'].item()
    posDistSqThr = mat['dbStruct'][0, 0]['posDistSqThr'].item()
    if 'nonTrivPosDistSqThr' in fieldnames:
        nonTrivPosDistSqThr = mat['dbStruct'][0, 0]['nonTrivPosDistSqThr'].item()
    else:
        nonTrivPosDistSqThr = posDistSqThr
    if 'utmDb' in fieldnames and 'utmQ' in fieldnames:
        utmDb = mat['dbStruct'][0, 0]['utmDb'].T
        utmQ = mat['dbStruct'][0, 0]['utmQ'].T
    else:
        utmDb = None
        utmQ = None
    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, utmQ, numDb, numQ,
                    posDistThr, posDistSqThr, nonTrivPosDistSqThr, None, None, None, None)
