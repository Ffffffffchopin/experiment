from posixpath import split
import torch
import  numpy
import os
from operator import itemgetter
import codecs

#获取标签值的类别和数量
def num_class_Tensor(target_Tensor):
    target_Tensor=target_Tensor.numpy()
    class_dict={}
    for x in target_Tensor:
        if x not in class_dict.keys():
            class_dict[x]=0
        class_dict[x]+=1
    return class_dict


#得到新query的标签值
#def get_new_qureyed_y(pos_idx):
    

#将str写入文件
def writeto_file(content): 
    f=open('num_class.txt','a')
    f.writelines(content+'\n')
    f.writelines('\n')
    f.close()

#自动按照strategies执行主程序
def iter_get_num_class():
    strategy_list=["RandomSampling", 
                             "LeastConfidence", 
                             "MarginSampling", 
                             "EntropySampling", 
                             "LeastConfidenceDropout", 
                             "MarginSamplingDropout", 
                             "EntropySamplingDropout", 
                             "KMeansSampling",
                             "KCenterGreedy", 
                             "BALDDropout", 
                             "AdversarialBIM", 
                             "AdversarialDeepFool"]
    for strategy in strategy_list:
        f=open('num_class.txt','a')
        f.writelines(strategy+'\n')
        f.writelines('\n')
        f.close()
        cmd='python demo.py --n_init_labeled 100 --n_query 10 --strategy_name '+strategy
        os.system(cmd)

#得到新query的标签
def get_new_query(strategy,query_idxs):
    new_query_y=strategy.get_new_queryed_y(query_idxs)
    #content= num_class_Tensor(new_query_y)
    return new_query_y

def calc_max_dist(label_dict):
    sorteddict=sorted(label_dict.items(),key=itemgetter(1))
    return sorteddict[len(sorteddict)-1][1]-sorteddict[0][1]


def process_ImageNet_LT():
    image_path=[]
    label_path=[]
    image_tmp_path=[]
    label_tmp_path=[]
    #train和test分开
    for mode in ['train','test']:

    #得到写入文件
        with open('C:\\Users\\F.F.Chopin\\project\\low-budget-al\\ImageNet_LT_open\\{}_name.txt'.format(mode),'r') as f :
            for line in f.readlines():

    #获取图片路径列表
                image_tmp_path.append(str(line.split()[0]))
                label_tmp_path.append(int(line.split()[1]))
        image_path.append(image_tmp_path)
        label_path.append(label_tmp_path)  

    return numpy.array(image_path),numpy.array(label_path)

#获取二进制转换为hex格式
def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


if __name__=='__main__':
    #test_Tensor=torch.tensor((1,3,5,4,4))
    #print(test_Tensor)
    #print(num_class_Tensor(test_Tensor))
    #iter_get_num_class()  
    image_path,label_path= process_ImageNet_LT()
    print((label_path[1])) 


