'''
重复网上的一个例子
用keras 搭建LSTM，使用嵌入层实现文本向量
url: https://www.jianshu.com/p/795a5e2cd10c
by xlc time:2018-03-08 20:53:01
'''
import sys
sys.path.append('G:/Github_codes/mypyfunc')
import numpy as np
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation

if __name__ == '__main__':
    raw_data_path = r'G:\xlcFiles\files_data\20_newsgroup'
    glove_100_path = r"G:\xlcFiles\files_data\glove.6B\glove.6B.100d.txt"
    validation_percent = .2 #训练集占比

    #获得词向量
    word_vec = {}
    f = open(glove_100_path, 'r', encoding='utf-8')
    for line in f:
        string = line.strip().split(' ', 1)
        word = string[0]
        vec_str = string[1].split(' ')
        vec = [float(i) for i in vec_str]
        word_vec[word] = vec
    #获得训练文本
    texts = []
    labels = []
    label_index = {}
    text_folds = os.listdir(raw_data_path)
    for i in text_folds:
        now_path = raw_data_path + '/' + i#获得文本的当前路径
        text_files = os.listdir(now_path) #获得当前路径下的文本
        text_files = [i for i in text_files if i.isdigit()]
        index = text_folds.index(i) #获得文件夹的索引
        label_index[i] = index
        for txt in text_files:
            text = open(now_path + '/' + txt, 'r', encoding='latin-1').read()
            texts.append(text)
            labels.append(index)

    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index

    data = pad_sequences(sequences, maxlen=1000)
    labels = to_categorical(np.asarray(labels))

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    split_point = int(0.2 * data.shape[0])

    X_train = data[:-split_point]
    y_train = labels[:-split_point]

    X_valid = data[-split_point:]
    y_valid = labels[-split_point:]

    #整理词向量
    #根据上上步得到每一个词的权重（词频）
    #然后根据得到的编号，与Glove词向量对应起来
    min_split = min(20000, len(word_index))
    embedding_matrix = np.zeros((min_split+1, 100))
    for word, idx in word_index.items():
        if idx > 20000:
            continue
        else:
            embedding_vec = word_vec.get(word)
            if embedding_vec is not None:
                embedding_matrix[idx] = embedding_vec
    
    #开始进行LSTM 训练
    embedding_layer = Embedding(20001,
                                100,
                                weights=[embedding_matrix],
                                input_length=1000,
                                dropout=0.2)
    print('开始建立模型')
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.add(Dense(len(label_index), activation='softmax'))
    model.layers[1].trainable=False

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, nb_epoch=5,
              validation_data=(X_valid, y_valid))
    score, acc = model.evaluate(X_valid, y_valid,
                                batch_size=32)
    
    
