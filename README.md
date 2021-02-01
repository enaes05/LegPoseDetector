# AlphaPose

## Instalación
```shell
# 1. Instalar PyTorch
pip3 install torch==1.1.0 torchvision==0.3.0

# 2. Obtener AlphaPose
git clone https://github.com/enaes05/AlphaPose.git
cd AlphaPose

# 3. Instalar
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
pip install cython
sudo apt-get install libyaml-dev
python setup.py build develop --user
```

### Preparación
1. Para descargar el detector de objetos: **yolov3-spp.weights**([Google Drive](https://drive.google.com/open?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC) | [Baidu pan](https://pan.baidu.com/s/1Zb2REEIk8tcahDa8KacPNA)). Se coloca en `detector/yolo/data`.

2. Para descargar el identificador de pose: **JDE-1088x608-uncertainty**([Google Drive](https://drive.google.com/open?id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA) | [Baidu pan](https://pan.baidu.com/s/1Ifgn0Y_JZE65_qSrQM2l-Q)). Se coloca en `detector/tracker/data`.

3. Se pueden descargar modelos ya entrenados. Para ello se puede consultar la documentación original de AlphaPose, en https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md

## Cómo entrenar
```python3 ./scripts/train.py --cfg path_to_config_file --exp-id training_identifier```

## Cómo testear imágenes
```python3 ./scripts/demo_inference.py --cfg path_to_config_file --checkpoint path_to_checkpoint --save_img --detbatch 1 --posebatch 30 --image path_to_image```

Las imágenes se guardan por defecto en ```/examples/res/vis```. Para cambiar el directorio, se usa el parámeto ```--outdir directory```.

## Cómo testear vídeos
```python3 ./scripts/demo_inference.py --cfg path_to_config_file --checkpoint path_to_checkpoint --save_video --detbatch 1 --posebatch 30 --video path_to_video```

Los vídeos se guardan por defecto en ```/examples/res```. Para cambiar el directorio, se usa el parámeto ```--outdir directory```.
