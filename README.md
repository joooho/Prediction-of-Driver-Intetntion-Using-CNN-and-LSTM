# Prediction-of-Driver-Intetntion-Using-CNN-and-LSTM
## 프로젝트 설명
1. "도로에서 운전자의 의도를 예측을 할 수 있다면 교통 사고를 예방할 수 있을 것이다"라는 주제로 시작하게 된 프로젝트입니다.
2. 교통 사고 분석 시스템(TAAS)의 통계에 따르면 도로형태별 기준, 교차로 사고가 49.8%로 단일로 사고 45.7%보다 더 많이 발생함을 알 수 있습니다. 
3. 따라서, 저희는 교차로에서의 운전자 의도 상태를 총 4가지의 경우로 구별하였습니다. 

교차로 부근에서 속도를 낮추다가 정지선을 지나치며 속도를 올려 증가하기 시작할 때의 상태를 직진(Go),

차량이 교차로 부근해서 속도를 낮추고 속도가 0이 되었을 때의 상태를 정지(Stop),

차량이 교차로 부근에서 운전대를 왼쪽으로 돌려(+)값이 나올 때의 상태를 좌회전(Left),

차량이 교차로 부근에서 운전대를 오른쪽으로 돌려(-)값이 나올 때의 상태를 우회전(Right)이라 구분하였습니다.


## 선행조건
- Pytorch
- Pandas (커스텀 데이터 세트 Load)
- Sklearn (전처리 및 데이터 분할)
- visdom (train 및 test 결과 시각화)

## 기타사항
- 본 프로젝트에 사용된 데이터는 외부에서 구입한 데이터로 배포할 수 없습니다.
- 모델의 Input Data는 각종 센서로부터 받아온 시계열 데이터가 Pandas, Sklearn, torch.util.data.Dataload를 통해 (Batch size, Time step, Feature)로 변환됩니다.

## 개발인원
1. 정주호
2. 방건호

## 참고자료
- https://dzlab.github.io/timeseries/2018/11/25/LSTM-FCN-pytorch-part-1/
- Karim, F., Majumdar, S., Darabi, H., & Chen, S. (2017). LSTM fully convolutional networks for time series classification. IEEE access, 6, 1662-1669.
