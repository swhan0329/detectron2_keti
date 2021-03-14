# [100] 교통문제 해결을 위한 CCTV 교통 데이터(고속도로) AI 모델

## 결과물

* 고속도로 CCTV 영상 50개, 500시간 이상 영상
* 차량 검출 30 만장, 차량 분할 20 만장 이미지(png) 및 어노테이션 파일(json)
* 속도 추정 GT 데이터로 사용하기 위한 VDS 영상 10시간

## 객체 검출, 분할 모델

본 과제에서 구축한 차량 검출, 분할 데이터셋 검증을 위해 객체 검출, 분할 모델을 사용

### Reference code

#### Detectron2

* Detectron2는 최첨단 객체 감지 알고리즘을 구현하는 Facebook AI Research의 차세대 소프트웨어 시스템 

* 이전 버전인 [Detectron](https://github.com/facebookresearch/Detectron/)을 완전히 재작성한 것으로 [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/)를 기반으로 모델을 구성함

#### Detectron2 official code

Facebook AI Research에서 만든 Detectron2 공식 GitHub 주소

https://github.com/facebookresearch/detectron2

## 사용방법

데이터셋, 모델웨이트, 도커 등 세팅 방법은 아래 링크 참조
