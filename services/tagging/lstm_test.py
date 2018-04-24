import numpy as np
from keras.layers import Dropout, Dense
from keras.layers import GaussianNoise
from keras.layers import LSTM
from keras.models import Sequential
from scipy import spatial

import services.tagging.preprocessing as preprocessing
import services.tagging.featureextraction as featureextraction

batch_size = 1
samples = 1




featureextraction.create_wordvec()




def createNetwork(x_train,y_train):
    global model
    model = Sequential()
    model.add(GaussianNoise(0.1,input_shape=(None,300)))
    model.add(LSTM(32,input_shape=(None,300),return_sequences=False,batch_size=batch_size))
    model.add(Dropout(0.5))
    model.add(Dense(300))
    model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['accuracy']) #cosine proximity
    print(model.summary())


def prepare_train_sets():
    x_train, y_train,x_test,y_test = preprocessing.get_data()

    test_text = []
    test_tag = []

    for idx, text in enumerate(x_train):
        x_train[idx] = np.array(
            featureextraction.get_vecs(preprocessing.preprocessing_pipeline(text, featureextraction.wordvecs)))
        x_train[idx] = np.reshape(x_train[idx],(1,len(x_train[idx]),300))
        print(x_train[idx].shape)
    #x_train = np.array(x_train)


    for idx,tag in enumerate(y_train):
        y_train[idx]=np.array(featureextraction.get_vecs(tag))

    for idx, text in enumerate(x_test):
        test_text.append(str(x_test[idx]))
        print(test_text[idx])
        x_test[idx] = np.array(
            featureextraction.get_vecs(preprocessing.preprocessing_pipeline(text, featureextraction.wordvecs)))
        x_test[idx] = np.reshape(x_test[idx],(1,len(x_test[idx]),300))

    for idx,tag in enumerate(y_test):
        test_tag.append(str(tag))
        print(test_text[idx])
        y_test[idx]=np.array(featureextraction.get_vecs(tag))

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train,y_train,x_test,y_test,test_text,test_tag


def get_accuracy(x_test,y_test):
    expected_wordvec = y_test
    prediction = model.predict(x_test)
    result = 1 - spatial.distance.cosine(expected_wordvec,prediction)


    print("SIMILARITY: ",result)

    prediction = np.array(prediction)
    prediction=np.reshape(prediction,prediction.shape[1])
    print(featureextraction.wordvecs.similar_by_vector(prediction))




x_train,y_train,x_test,y_test,test_text,test_tag = prepare_train_sets()
print(len(x_train))
createNetwork(x_train[0],y_train[0])


for i in range(0,len(x_train)):
    print(x_train[i].shape)
    y_train[i] = y_train[i].reshape(1,1,300)
    model.train_on_batch(x_train[i],y_train[i])

    print("batch trained: ",i)
print(len(test_tag))

for i in range(0,len(x_test)):
    y_test[i] = y_test[i].reshape(1,1,300)
    print("TEST SAMPLE: ",i)
    print(test_text[i])
    print("TAG: ",test_tag[i])
    get_accuracy(x_test[i],y_test[i])
    print(model.test_on_batch(x_test[i],y_train[i]))




#x_train[0]=x_train[0].reshape(1,96,300)
#np.reshape(y_train[0],(1,1,300))
#createNetwork(x_train[0],y_train[0])




