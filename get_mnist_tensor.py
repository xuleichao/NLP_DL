'''

by xlc time:2018-03-30 22:06:00
'''
import sys
sys.path.append('G:/Github_codes/mypyfunc')
import numpy as np
import struct

def loadImageSet(which=0):
    print("load image set")
    binfile=None
    if which==0:
        binfile = open("train-images-idx3-ubyte/train-images.idx3-ubyte", 'rb')
    else:
        binfile=  open("t10k-images-idx3-ubyte/t10k-images.idx3-ubyte", 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII' , buffers ,0)
    print("head,",head)

    offset=struct.calcsize('>IIII')
    imgNum=head[1]
    width=head[2]
    height=head[3]
    #[60000]*28*28
    bits=imgNum*width*height
    bitsString='>'+str(bits)+'B' #like '>47040000B'

    imgs=struct.unpack_from(bitsString,buffers,offset)

    binfile.close()
    imgs=np.reshape(imgs,[imgNum,width,height])
    print("load imgs finished")
    return imgs

def loadLabelSet(which=0):
    print("load label set")
    binfile=None
    if which==0:
        binfile = open("train-labels-idx1-ubyte/train-labels.idx1-ubyte", 'rb')
    else:
        binfile=  open("t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte", 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II' , buffers ,0)
    print("head,",head)
    imgNum=head[1]

    offset = struct.calcsize('>II')
    numString='>'+str(imgNum)+"B"
    labels= struct.unpack_from(numString , buffers , offset)
    binfile.close()
    labels=np.reshape(labels,[imgNum,1])

    #print labels
    print('load label finished')
    return labels

#count 为batch 数量
def next_x_y(Image, Label, count): #图像和标签的next 方法，用于feed
    lower_shape = lambda x, y: y if x >= y else x 
    Image_shape = Image.shape[0]
    Label_shape = Label.shape[0]
    lwr_num = lower_shape(Image_shape, Label_shape)
    if lwr_num % count != 0:
        the_count = lwr_num // count
        print('输入的规模与所返回的规模不符，将迭代的个数为: ', the_count)

    else:
        the_count = lwr_num // count
    for i in range(the_count):
        b = i * count
        e = b + count
        yield Image[b:e], Label[b:e]

if __name__!="__main__":
    imgs=loadImageSet()
    #import PlotUtil as pu
    #pu.showImgMatrix(imgs[0])
    lbls = loadLabelSet()
