# 객체 검출, 분할 모델

## 모델 및 데이터셋 다운로드

1. 소스코드 다운로드

```bash
git clone https://github.com/swhan0329/detectron2_keti/
```

2. 다운로드

* 모델 웨이트

* 데이터셋

AI HUB에 올라와있는 (detection, segmentation) test 데이터셋 다운

* 도커이미지



3. 경로 설정

아래 코드 트리를 보고 경로를 세팅

* 객체 검출/분할 코드 트리

```
.
├── datasets
│   ├── test_BB
│   ├── test_BB10.json
│   ├── train_BB
│   ├── train_BB50.json
│   ├── test_PS
│   ├── test_PS10.json
│   ├── train_PS
│   └── train_PS50.json
├── detectron2_keti
│   ├── build
│   ├── configs
│   ├── demo
│   ├── detectron2
│   ├── detectron2.egg-info
│   ├── dev
│   ├── docker
│   ├── docs
│   ├── LICENSE
│   ├── MODEL_ZOO.md
│   ├── projects
│   ├── README.md
│   ├── setup.cfg
│   ├── setup.py
│   ├── tests
│   └── tools
└── weights
    └── model_final_PS.pth 
    └── model_final_BB.pth 
```

## 도커 이미지 사용 매뉴얼

1. Docker 이미지 로드

```bash
docker load -i detectron2-keti.tar
```

2. Docker 컨테이너 생성

* source: 코드 및 데이터셋이 있는 폴더

```bash
docker run --gpus all -it --name=detectron2 --mount type=bind,source=/home/super/sw/100,target=/home/appuser detectron2:v0
```

3. detectron2_keti 폴더로 이동

```bash
cd detectron2_keti
```

4. 코드 내 C 코드 컴파일

```bash
python setup.py build develop
```

5. detectron2 폴더로 이동

```bash
cd detectron2
```

6. train/test 스크립트 실행

```bash
train_BB.py / train_PS.py / test_BB.py / test_PS.py
```
