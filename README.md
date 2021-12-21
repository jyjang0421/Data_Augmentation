# Data_Augmentation
## OpenSource Project
## 1. 문제 정의

   - 사진을 이용하여 폐렴인지 아닌지 구별하는 문제를 ResNetV2 모델을 이용하여 해결하고자 한다. 
 
   - 하지만, 제가 구한 데이터의 수가 충분하지 못하여 우리가 구현한 모델을 제대로 training하지 못하고, 그에 따라 모델을 generalization을 하지 못하는 문제를 겪을 것으로 예상된다. 
 
   - 이 문제를 해결하기 위해 우리는 Data Augmentation 방법을 수행해 데이터를 충분히 확보하여 Model을 Generalization하고자 한다. 
  
## 2. Dataset 설명

   - Train data(Imbalance Dataset) 
     - Pneumonia : 3875개 
     - Normal : 1341개
 
   - Validation data 
     - Pneumonia : 8개, Normal : 8개
 
   - Test data
     - Pneumonia : 390개, Normal : 234개


## 3. 각각의 파일에 대한 설명
   
 - original.ipynb(Data Augmentation을 적용하지 않은 데이터로 Resnet모델 학습)
   - 참고자료 : (https://www.kaggle.com/sauravbhuyan/pneumonia-classification-with-chestxray)

   - 학습 방법 :
     - Resnet모델의 imagenet weight값을 가져와 제 모델에 적용시키고 추가로 제 task에 맞게GlobalAvgPooling, Dense layer를 추가 시킵니다.
     - 제가 추가한 layer만 학습시키고 나머지 layer는 학습시키지 않습니다.
     - **더 나아가 기존에서 제시한 방법보다 더 성능을 향상시키기 위해** Validation accuracy가 어느 정도 향상된 후, 뒤에서 5개의 layer를 추가로 학습 시킵니다.(Fine Tuning) 
   - 학습 결과 : 
     - Load Model의 layer전체를 freeze한 후 학습한 결과
      -> Test Data Accuracy : 71.31%
     - Fine Tuning한 후 학습 결과
      -> Test Data Accuracy : 73.56%
     - 앞의 모델보다 2%정도 모델의 성능이 좋아진 것을 확인할 수 있다.

   - 추가 연구 :
     - 원본 이미지와 rescaling한 이미지 와의 차이점을 확인하여 (224,224,3)으로 줄여도 되는지 확인
     - Rotation_range =10으로 적용하여 augmentation된 이미지 확인하기



### Accuracy & Loss Plot        
<img width="400" alt="그림1" src="https://user-images.githubusercontent.com/60686561/146888973-1937b394-a902-43c8-8522-e339d57cd1e2.png"><img width="409" alt="그림2" src="https://user-images.githubusercontent.com/60686561/146888984-25dd97b5-542d-4cc0-ac02-aab715dc0fc8.png">



### Original image vs Rescaled image
<img width="400" alt="그림7"  src="https://user-images.githubusercontent.com/60686561/146891420-f9afabff-4918-4a89-840a-3ea4fc150266.png"><img width="423" alt="그림8" src="https://user-images.githubusercontent.com/60686561/146891428-c866a3ea-5e3b-40ee-a659-3ad3d6bc765e.png">


### Before vs After Augmentation
<img width="300" alt="그림9" src="https://user-images.githubusercontent.com/60686561/146891433-b79c44f3-5901-423d-a24f-5eec66966b34.png"><img width="400" alt="그림10" src="https://user-images.githubusercontent.com/60686561/146891435-64d96835-b81d-4bc8-9420-f665465fb7ca.png">


 - data_augment1.ipynb(Data Augmentation을 적용한 데이터로 Resnet모델 학습)
   - Dataset 구성 
     - Train data의 normal data 와 abnoraml data 구성을 5000개씩 1:1로 증강시킨다.
     - Train data를 split하여 train data와 validation data를 나눈다.(8:2)
     - Augmentation parameter 적용 
       - Rotation_range = 10
       - Shear_range = 0.2
       - Width_shift_range = 0.05
       - Height_shift_range = 0.05
       - Zoom_range = 0.2
       - Rescale = 1/255

   - 학습 결과 
     - Load Model의 layer전체를 freeze한 후 학습한 결과
       -> Test Data Accuracy : 86.38%
     - Fine Tuning한 후 학습 결과
       -> Test Data Accuracy : 86.70%
     - 앞의 모델보다 0.4%정도 모델의 성능이 좋아진 것을 확인할 수 있다.
### Accuracy & Loss Plot  
<img width="410" alt="그림3" src="https://user-images.githubusercontent.com/60686561/146888991-624e1b5d-ed3e-4ff2-be94-476331be9931.png"><img width="417" alt="그림4" src="https://user-images.githubusercontent.com/60686561/146888993-d87b501e-d4c7-43c3-9b4a-2b22052de619.png">

   - data_augment2.ipynb(Data Augmentation을 적용한 Resnet데이터로 모델 학습)
       - Dataset 구성
          - Augmentation Parameter 적용 
          - Rotation_range = 10
          - Shear_range = 0.1
          - Width_shift_range = 0.1
          - Height_shift_range = 0.1
          - Zoom_range = 0.1
          - Rescale = 1/255
      - 학습 결과
        - Load Model의 layer전체를 freeze한 후 학습한 결과
          -> Test Data Accuracy : 84.78%
        - Fine Tuning한 후 학습 결과
          -> Test Data Accuracy : 86.54%
        - 앞의 모델보다 약 2%정도 모델의 성능이 좋아진 것을 확인할 수 있다. 
        
### Accuracy & Loss Plot          
<img width="417" alt="그림5" src="https://user-images.githubusercontent.com/60686561/146888995-8eabb1ef-f1fd-4216-ba88-400a77edbb52.png"><img width="413" alt="그림6" src="https://user-images.githubusercontent.com/60686561/146888998-94516be1-3516-4aa9-8454-1d80c49c0258.png">

## 4.결과
   - 데이터 증강 전 가장 높은 test accuracy : 73.56%
   - 데이터 증강 후 가장 높은 test accuracy : 86.70%
   - Data Augmentation을 적용하지 않은 model보다 적용한 model의 accuracy가 **13.24%**  증가한 것을 확인하였다.
   
