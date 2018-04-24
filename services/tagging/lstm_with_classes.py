import numpy as np
from keras.layers import GaussianNoise
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from keras.models import model_from_json

import services.tagging.featureextraction as featureextraction
import services.tagging.preprocessing as preprocessing
from services.tagging.laplotter import LossAccPlotter

plotter = LossAccPlotter()


def main():
    create_Network()
    featureextraction.create_wordvec()
    x_train, y_train, x_test, y_test,tag_train,tag_test,test_text = prepare_input()
    feed_input(x_train, y_train, x_test, y_test,tag_train,tag_test,test_text)


def create_Network():

    #model1
    #global model
    #model = Sequential()
    #model.add(GaussianNoise(0.3,input_shape=(None,300)))
    #model.add(LSTM(32,input_shape=(None,300),return_sequences=True,batch_size=1))
    #model.add(Dropout(0.3))
    #model.add(LSTM(32,input_shape=(None,300),return_sequences=False,batch_size=1))
    #model.add(Dropout(0.3))
    #model.add(Dense(10,activation='softmax'))


    #model2
    #global model
    #model = Sequential()
    #model.add(GaussianNoise(0.3, input_shape=(None, 300)))
    #model.add(LSTM(32, input_shape=(None, 300), return_sequences=False, batch_size=1))
    #model.add(Dropout(0.3))
    #model.add(Dense(10, activation='softmax'))



    # model3
    global model
    model = Sequential()
    model.add(GaussianNoise(0.1,input_shape=(None,300)))
    model.add(LSTM(32,input_shape=(None,300),return_sequences=True,batch_size=1))
    model.add(Dropout(0.3))
    model.add(LSTM(32,input_shape=(None,300),return_sequences=False,batch_size=1))
    model.add(Dropout(0.3))
    model.add(Dense(10,activation='softmax'))

    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    print(model.summary())

def prepare_input():
    x_train, y_train, x_test, y_test, tag_train, tag_test = preprocessing.get_data_withOneHot(10, 5)
    test_text = []
    print("Creating Wordvecs")

    for idx,text in enumerate(x_train):
        x_train[idx] = np.array(
            featureextraction.get_vecs(preprocessing.preprocessing_pipeline(text, featureextraction.wordvecs)))
        x_train[idx] = x_train[idx].reshape(1,len(x_train[idx]),300)
    for idx,text in enumerate(x_test):
        test_text.append(x_test[idx])
        x_test[idx] = np.array(
            featureextraction.get_vecs(preprocessing.preprocessing_pipeline(text, featureextraction.wordvecs)))
        x_test[idx] = x_test[idx].reshape(1, len(x_test[idx]), 300)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_train=y_train.reshape(y_train.shape[0],1,10)

    print(np.shape(y_train[0]))
    y_test = y_test.reshape(y_test.shape[0],1,10)
    print("Creating Wordvecs --> FINISHED")

    return x_train,y_train,x_test,y_test,tag_train,tag_test,test_text


def feed_input(x_train,y_train,x_test,y_test,tag_train,tag_test,test_text):
    print("feeding input")

    #50 epochs --> 56%
    #150 epochs -->
    for epoch in range(120):
        for i in range(0,len(x_train)):
            print("Training Batch ",i," of epoch ",epoch)
            loss_train,acc_train=model.train_on_batch(x_train[i],y_train[i])
            #plotter.add_values(epoch,loss_train=loss_train)

    prediction_correct = 0
    test_samples = 0

    for i in range(0,len(x_test)):
        print("\n \n \n \n \n \n")
        print("Testing Batch - ",i)
        print(test_text[i])
        loss_test,acc_test = model.test_on_batch(x_test[i],y_test[i])
        prediction = model.predict(x_test[i])
        print(loss_test,acc_test)
        test_samples += 1
        print("Expected: ",tag_test[i])
        print("Predicted Tag: ", preprocessing.taglist[np.argmax(prediction)])
        print("Predicted: ",prediction)
        if acc_test == 1.0:
            prediction_correct += 1
        plotter.add_values(i,loss_test,acc_test)


    accuracy = prediction_correct/test_samples
    accuracy = accuracy*100
    print("feeding input --> FINISHED")
    print("ACCURACY: ",accuracy,"%")

    print("Saving Model")
    save_model(model)



def lstm_prediction(text,threshold):
    x_predict = np.array(
        featureextraction.get_vecs(preprocessing.preprocessing_pipeline(text, featureextraction.wordvecs)))
    x_predict = x_predict.reshape(1,len(x_predict),300)
    print("Loading Model")
    #loaded_model = load_model()
    prediction = loaded_model.predict(x_predict)
    print(prediction[0])
    for idx,element in enumerate(prediction[0]):
        if threshold/100 <= element:
            print(preprocessing.taglist[idx])


def save_model(model):
    model_json = model.to_json()
    with open("model4.json","w") as json:
        json.write(model_json)
    model.save_weights("model4.h5")
    print("Saving Model --> FINISHED")



def load_model():
    json = open('model3.json', 'r')
    loaded_json = json.read()
    json.close()
    loaded_model = model_from_json(loaded_json)
    loaded_model.load_weights("model3.h5")
    print("Loading Model --> FINISHED")
    return loaded_model


def make_prediction():
    print("Please Enter a Text")
    text = input()
    print("Please Enter a Threshold value")
    threshold = int(input())
    lstm_prediction(text, threshold)

main()
featureextraction.create_wordvec()
loaded_model = load_model()
while True:
   make_prediction()

plotter.block()