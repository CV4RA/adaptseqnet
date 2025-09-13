## Run

### Train
To train sequential descriptors through dsdnet on the Nordland dataset:
```python
python main.py --mode train --pooling adapnet --dataset nordland-sw --seqL 10 --w 5 --outDims 8192 --expName "w5"
```
or the Oxford dataset (set `--dataset oxford-pnv` for pointnetvlad-like data split:
```python
python main.py --mode train --pooling adapnet --dataset oxford-v1.0 --seqL 5 --w 3 --outDims 8192 --expName "w3"
```
or the MSLS dataset (specifying `--msls_trainCity` and `--msls_valCity` as default values):
```python
python main.py --mode train --pooling adapnet --dataset msls --msls_trainCity melbourne --msls_valCity austin --seqL 5 --w 3 --outDims 8192 --expName "msls_w3"
```

To train transformed single descriptors through dsdnet:
```python
python main.py --mode train --pooling adapnet --dataset nordland-sw --seqL 1 --w 1 --outDims 8192 --expName "w1"
```

### Test
On the Nordland dataset:
```python
python main.py --mode test --pooling adapnet --dataset nordland-sf --seqL 5 --split test --resume ./data/runs/Jun03_15-22-44_l10_w5/ 
```
On the MSLS dataset (can change `--msls_valCity` to `melbourne` or `austin` too):
```python
python main.py --mode test --pooling adapnet --dataset msls --msls_valCity amman --seqL 5 --split test --resume ./data/runs/<modelName>/
```
  
## Acknowledgement
The code in this repository is based on [seqNet](https://github.com/oravus/seqNet).