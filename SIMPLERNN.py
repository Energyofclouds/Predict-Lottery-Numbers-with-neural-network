
import numpy as np

rows = np.loadtxt("C:/Users/빈운기/Downloads/lotto.csv", delimiter=",")

    
row_count = len(rows)
print(row_count)
import numpy as np

# 당첨번호를 원핫인코딩벡터(ohbin)으로 변환
def numbers2ohbin(numbers):

    ohbin = np.zeros(45) #45개의 빈 칸을 만듬

    for i in range(6): #여섯개의 당첨번호에 대해서 반복함
        ohbin[int(numbers[i])-1] = 1 #로또번호가 1부터 시작하지만 벡터의 인덱스 시작은 0부터 시작하므로 1을 뺌
    
    return ohbin

# 원핫인코딩벡터(ohbin)를 번호로 변환
def ohbin2numbers(ohbin):

    numbers = []
    
    for i in range(len(ohbin)):
        if ohbin[i] == 1.0: # 1.0으로 설정되어 있으면 해당 번호를 반환값에 추가한다.
            numbers.append(i+1)
    
    return numbers
numbers = rows[:, 1:7]
ohbins = list(map(numbers2ohbin, numbers))

x_samples = ohbins[0:row_count-1]
y_samples = ohbins[1:row_count]

train_idx = (0, 800)
val_idx = (801, 900)
test_idx = (901, len(x_samples))


def gen_numbers_from_probability(nums_prob): # 볼추첨함수

    ball_box = []

    for n in range(45):
        ball_count = int(nums_prob[n] * 100 + 1)
        ball = np.full((ball_count), n+1) #1부터 시작
        ball_box += list(ball)

    selected_balls = []

    while True:
        
        if len(selected_balls) == 6:
            break
        
        ball_index = np.random.randint(len(ball_box), size=1)[0]
        ball = ball_box[ball_index]

        if ball not in selected_balls:
            selected_balls.append(ball)

    return selected_balls





print(len(x_samples))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN

xs=np.zeros(((train_idx[1]-train_idx[0]), 1,45))
ys=np.zeros((((train_idx[1]-train_idx[0]),45)))

xs_v=np.zeros(((val_idx[1]-val_idx[0]+1), 1,45))
ys_v=np.zeros(((val_idx[1]-val_idx[0]+1),45))

xs_t=np.zeros(((test_idx[1]-test_idx[0]+1), 1,45))
ys_t=np.zeros(((test_idx[1]-test_idx[0]+1),45))

xs_tt=np.zeros((len(x_samples), 1,45))
ys_tt=np.zeros((len(x_samples),45))

for i in range(0, test_idx[1]):
    xs_tt[i] = x_samples[i].reshape(1, 1, 45)
    ys_tt[i] = y_samples[i].reshape(1, 45)



for i in range(train_idx[0], train_idx[1]):
    xs[i] = x_samples[i].reshape(1, 1, 45)
    ys[i] = y_samples[i].reshape(1, 45)

for i in range(val_idx[0]-1, val_idx[1]):
    xs_v[i-val_idx[0]+1] = x_samples[i].reshape(1, 1, 45)
    ys_v[i-val_idx[0]+1] = y_samples[i].reshape(1, 1, 45)

for i in range(test_idx[0]-1, test_idx[1]):
    xs_t[i-test_idx[0]+1] = x_samples[i].reshape(1, 1, 45)
    ys_t[i-test_idx[0]+1] = y_samples[i].reshape(1, 45)
    
# 모델을 정의합니다.
model=Sequential()
model.add(SimpleRNN(128, batch_input_shape=(1, 1, 45)))
model.add(Dense(45, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history= model.fit(xs,ys,epochs=2)
model.summary()
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

mean_prize = [ np.mean(rows[87:, 8]),
           np.mean(rows[87:, 9]),
           np.mean(rows[87:, 10]),
           np.mean(rows[87:, 11]),
           np.mean(rows[87:, 12])]

print(mean_prize)   



def calc_reward(true_numbers, true_bonus, pred_numbers):

    count = 0

    for ps in pred_numbers:
        if ps in true_numbers:
            count += 1

    if count == 6:
        return 0, mean_prize[0]
    elif count == 5 and true_bonus in pred_numbers:
        return 1, mean_prize[1]
    elif count == 5:
        return 2, mean_prize[2]
    elif count == 4:
        return 3, mean_prize[3]
    elif count == 3:
        return 4, mean_prize[4]

    return 5, 0



train_total_reward = []
train_total_grade = np.zeros(6, dtype=int)

val_total_reward = []
val_total_grade = np.zeros(6, dtype=int)

test_total_reward = []
test_total_grade = np.zeros(6, dtype=int)


print('[No. ] 1st 2nd 3rd 4th 5th 6th Rewards')

for i in range(len(x_samples)):
    ys_pred = model.predict(xs_tt[i].reshape(1, 1, 45)) # 모델의 출력값을 얻음
    
    sum_reward = 0
    sum_grade = np.zeros(6, dtype=int) # 6등까지 변수

    for n in range(10): # 10판 수행
        numbers = gen_numbers_from_probability(ys_pred[0])
        
        #i회차 입력 후 나온 출력을 i+1회차와 비교함
        grade, reward = calc_reward(rows[i+1,1:7], rows[i+1,7], numbers) 
       
        sum_reward += reward
        sum_grade[grade] += 1

        if i >= train_idx[0] and i < train_idx[1]:
            train_total_grade[grade] += 1
        elif i >= val_idx[0] and i < val_idx[1]:
            val_total_grade[grade] += 1
        elif i >= test_idx[0] and i < test_idx[1]:
            test_total_grade[grade] += 1
    
    if i >= train_idx[0] and i < train_idx[1]:
        train_total_reward.append(sum_reward)
    elif i >= val_idx[0] and i < val_idx[1]:
        val_total_reward.append(sum_reward)
    elif i >= test_idx[0] and i < test_idx[1]:
        test_total_reward.append(sum_reward)
                        
    #print('[{0:4d}] {1:3d} {2:3d} {3:3d} {4:3d} {5:3d} {6:3d} {7:15,d}'.format(i+2, sum_grade[0], sum_grade[1], sum_grade[2], sum_grade[3], sum_grade[4], sum_grade[5], int(sum_reward)))

print('Total') 
print('===========================================================================================================================================')    
print('Train(800회 X 10 = 8000번)  1등 : {0:5d} / 2등 : {1:5d} / 3등 : {2:5d} / 4등 : {3:5d} / 5등 : {4:5d} / 낙첨 : {5:5d} / 총상금 : {6:15,d}'.format(train_total_grade[0], train_total_grade[1], train_total_grade[2], train_total_grade[3], train_total_grade[4], train_total_grade[5], int(sum(train_total_reward))))
print('Val(100회 X 10 = 1000번)    1등 : {0:5d} / 2등 : {1:5d} / 3등 : {2:5d} / 4등 : {3:5d} / 5등 : {4:5d} / 낙첨 : {5:5d} / 총상금 : {6:15,d}'.format(val_total_grade[0], val_total_grade[1], val_total_grade[2], val_total_grade[3], val_total_grade[4], val_total_grade[5], int(sum(val_total_reward))))
print('Test(91회 X 10 = 910번)     1등 : {0:5d} / 2등 : {1:5d} / 3등 : {2:5d} / 4등 : {3:5d} / 5등 : {4:5d} / 낙첨 : {5:5d} / 총상금 : {6:15,d}'.format(test_total_grade[0], test_total_grade[1], test_total_grade[2], test_total_grade[3], test_total_grade[4], test_total_grade[5], int(sum(test_total_reward))))
print('===========================================================================================================================================')

train_count=(train_idx[1]-train_idx[0])*10
val_count=(val_idx[1]-val_idx[0]+1)*10
test_count=(test_idx[1]-test_idx[0]+1)*10


print('=============================================================================================================================')    
print('Train_set 당첨 비율    1등 : {0:5.2f}% 2등 : {1:5.2f}% 3등 : {2:5.2f}% 4등 : {3:5.2f}% 5등 : {4:5.2f}% 낙첨 : {5:5.2f}%'.format(train_total_grade[0]/train_count*100, train_total_grade[1]/train_count*100, train_total_grade[2]/train_count*100, train_total_grade[3]/train_count*100, train_total_grade[4]/train_count*100, train_total_grade[5]/train_count*100))
print('Val_set 당첨 비율      1등 : {0:5.2f}% 2등 : {1:5.2f}% 3등 : {2:5.2f}% 4등 : {3:5.2f}% 5등 : {4:5.2f}% 낙첨 : {5:5.2f}%'.format(val_total_grade[0]/val_count*100, val_total_grade[1]/val_count*100, val_total_grade[2]/val_count*100, val_total_grade[3]/val_count*100, val_total_grade[4]/val_count*100, val_total_grade[5]/val_count*100))
print('Test_set 당첨 비율     1등 : {0:5.2f}% 2등 : {1:5.2f}% 3등 : {2:5.2f}% 4등 : {3:5.2f}% 5등 : {4:5.2f}% 낙첨 : {5:5.2f}%'.format(test_total_grade[0]/test_count*100, test_total_grade[1]/test_count*100, test_total_grade[2]/test_count*100, test_total_grade[3]/test_count*100, test_total_grade[4]/test_count*100, test_total_grade[5]/test_count*100))
print('=============================================================================================================================')    




#회차별 상금 그래프
import matplotlib.pyplot as plt

total_reward = train_total_reward + val_total_reward + test_total_reward

plt.plot(total_reward)
plt.ylabel('rewards')
plt.show()



# 데이터셋별로 상금 그래프
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

rewards = [sum(train_total_reward), sum(val_total_reward), sum(test_total_reward)]

class_color=['green', 'blue', 'red']

plt.bar(['train', 'val', 'test'], rewards, color=class_color)
plt.ylabel('rewards')
plt.show()


model1=Sequential()
model1.add(SimpleRNN(128, batch_input_shape=(1, 1, 45), return_sequences=False, stateful=True))
model1.add(Dense(45, activation='sigmoid'))

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history= model1.fit(xs_tt,ys_tt,epochs=300)


ys_last=ys_tt[len(x_samples)-1].reshape(1, 1,45)
predict_y_last=model1.predict(ys_last)
predict_y_last_list=list(predict_y_last)

predict_y_last_list_sort=[]
for i in range(0,45):
    predict_y_last_list_sort.append(predict_y_last_list[0][i])

predict_y_last_list_sort.sort()
print()
print(predict_y_last_list_sort)
predict_num=[]
cutter=predict_y_last_list_sort[39]






for i in range(0,45):
    if predict_y_last_list[0][i] >= cutter :
        predict_num.append(i+1)
        
    
        
        
        


model.summary()

print(" ")
print("992회차 예측 번호")
print(predict_num)
print(" ")

list_numbers = []
for n in range(10):
    numbers = gen_numbers_from_probability(predict_y_last_list[0])
    numbers.sort()
    print('{0} : [ {1:3d}, {2:3d}, {3:3d}, {4:3d}, {5:3d}, {6:3d}  ]'.format(n, numbers[0], numbers[1], numbers[2], numbers[3], numbers[4], numbers[5], ))
    list_numbers.append(numbers)

