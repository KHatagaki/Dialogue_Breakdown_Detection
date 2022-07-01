import os
import json
from sklearn.model_selection import train_test_split
import pandas as pd

def file_list(dir_name):
    file_list = []
    files = os.listdir(dir_name)
    for filename in files:
        if '.json' in filename:
            file_list.append(filename)
    return file_list

def make_dataset(from_path,to_path,files,num):
    if num == "0":
        file_name = "train.txt"
    else:
        file_name = "eval.txt"
    
    with open(to_path + file_name,"w") as g:
        for file in files:
            with open(from_path + file,"r") as f:
                jsn = json.load(f)
                for i,turns in enumerate(jsn['turns']):
                    cnt_O = 0
                    cnt_T = 0
                    cnt_X = 0
                    if(turns["annotations"] != [] and turns["speaker"] == "S"):
                        for annotation in turns["annotations"]:
                            if annotation["breakdown"] == "O":
                                cnt_O += 1
                            elif annotation["breakdown"] == "T":
                                cnt_T += 1
                            else:
                                cnt_X += 1
                        if (cnt_O > cnt_T and cnt_O > cnt_X):
                            label = "O"
                        elif (cnt_T > cnt_O and cnt_T > cnt_X):
                            label = "T"
                        elif (cnt_X > cnt_T and cnt_X > cnt_O):
                            label = "X"
                        elif (cnt_X == cnt_T):
                            label = "X"
                        elif (cnt_O == cnt_T):
                            label = "O"
                        else:
                            label = "T"
                        g.write(jsn['turns'][i-1]["utterance"].replace("\n","") + "|" + turns["utterance"].replace("\n","") + "|" +  label + "\n")

def split_data(to_path):
    path = to_path + "train.txt"
    df = pd.read_csv(path,sep='|',header=None)
    train,val = train_test_split(df,test_size=0.1,random_state=42)
    train.to_csv('./data/train.txt',sep='|',index=False,header=None)
    val.to_csv('./data/valid.txt',sep='|',index=False,header=None)

def main():
    train_dst = "/home/hatagaki/Dialog_System/breakdown/dataset/DBDC4_dev_20190312/en"
    eval_dst = "/home/hatagaki/Dialog_System/breakdown/dataset/DBDC4_eval_20200314/en"
    train_files = file_list(train_dst)
    eval_files = file_list(eval_dst)

    from_path = train_dst + "/"
    to_path = "/home/hatagaki/Dialog_System/breakdown/baseline/data/"
    make_dataset(from_path,to_path,train_files,"0")

    split_data(to_path)

    from_path = eval_dst + "/"
    to_path = "/home/hatagaki/Dialog_System/breakdown/baseline/data/"
    make_dataset(from_path,to_path,eval_files,"1")
    
    

if __name__ == "__main__":
    main()
