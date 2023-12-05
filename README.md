팀 프로젝트 때 사용한 데이터입니다.

https://www.kaggle.com/datasets/samyukthamurali/airbnb-ratings-dataset/download?datasetVersionNumber=12

1. Regression 모델 생성 결과
   
![image](https://github.com/UUUNG/BDP_Team_Project/assets/52543621/561720d4-b3af-4e2f-9e56-b82892aeb48d)

순서대로 

1. R-squared (R2) - 결정 계수
   - 모델이 주어진 데이터에 대해 얼마나 잘 적합되었는지를 나타내는 지표.
   - 값은 0에서 1 사이이며, 1에 가까울수록 모델이 데이터를 잘 설명한다는 것을 의미.
   - R2 값이 높을수록 모델이 데이터를 잘 예측한다고 판단할 수 있.
2. Mean Absolute Error (MAE) - 평균 절대 오차
   - 예측값과 실제 값의 차이를 절대값으로 변환하여 그 차이의 평균을 계산.
   - 값이 작을수록 모델의 예측이 실제 값과 가깝다는 것을 의미. 
3. Root Mean Squared Error (RMSE) - 평균 제곱근 오차
   - 예측값과 실제 값의 차이를 제곱한 후 평균을 구하고, 그 값의 제곱근을 구함.
   - RMSE 역시 작을수록 모델의 예측이 실제 값과 가깝다는 것을 의미.

2. Reagressio 모델 생성 결과(2)
   - 변수 분석 후 Review Scores Location 제외, Room Type 추가
   - ![image](https://github.com/UUUNG/BDP_Team_Project/assets/52543621/4c1474d4-198b-42eb-80a2-f9b5cffcf23c)
   - 0.5940271119420524, 47.28775292816959, 78.76177564874547로 성능이 더 좋아짐 (여전히 엄청 좋진 않음)
