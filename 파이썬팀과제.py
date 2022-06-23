from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm.notebook import tqdm
import csv
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


device = torch.device('cpu')

bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")


tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

## Setting parameters
max_len = 64
batch_size = 1
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)

model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

PATH = r"C:\Users\Admin\Desktop\team3\News.pt"

# 불러오기
device = torch.device('cpu')
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.to(device)
model.eval()


import pandas as pd

# 위에서 설정한 tok, max_len, batch_size, device를 그대로 입력
# comment : 예측하고자 하는 텍스트 데이터 리스트
def getSentimentValue(comment, tok, max_len, batch_size, device):
  commnetslist = [] # 텍스트 데이터를 담을 리스트
  emo_list = [] # 감성 값을 담을 리스트
  # for c in comment: # 모든 댓글
  commnetslist.append( [comment, 5] ) # [댓글, 임의의 양의 정수값] 설정
    
  pdData = pd.DataFrame( commnetslist, columns = [['댓글', '감성']] )
  pdData = pdData.values
  test_set = BERTDataset(pdData, 0, 1, tok, max_len, True, False) 
  test_input = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0)
  
  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_input):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length= valid_length 
    # 이때, out이 예측 결과 리스트
    out = model(token_ids, valid_length, segment_ids)
	
    # e는 2가지 실수 값으로 구성된 리스트
    # 0번 인덱스가 더 크면 부정, 긍정은 반대
    for e in out:
      if e[0]>e[1]: # 부정
        value = 0
      else: #긍정
        value = 1
      emo_list.append(value)

  return emo_list # 텍스트 데이터에 1대1 매칭되는 감성값 리스트 반환

from numpy import average, genfromtxt
import numpy as np

from selenium import webdriver
import time
from bs4 import BeautifulSoup
from urllib.request import urlopen, re

########### 전일 거래량 20종목 가져오기 ############

def getStocks():    
    # 주식수 (20개)
    stockNum = 20

    # 해당 페이지가 주식 정보를 동적으로 만들어 selenium으로 데이터를 가져온다.
    options = webdriver.ChromeOptions()
    options.add_argument('headless')

    driver = webdriver.Chrome(r'C:\Users\Admin\Desktop\team3\chromedriver.exe', options=options)


    ####################### 종목명, 종목정보 링크(etf종목 구별할 때 사용) #################

    driver.get('https://finance.naver.com/sise/sise_quant.naver')

    time.sleep(5)
    bs = BeautifulSoup(driver.page_source, 'html.parser')

    # 테이블에서 PER|ROE|영업이익(etf,etn면 N/A) 위치
    index = 0 
    ths = bs.find(class_='type_2').find('tbody').find('tr').find_all('th')
    for i in range(len(ths)):
        if(ths[i].get_text() == 'PER' or ths[i].get_text() == 'ROE' or ths[i].get_text() == '영업이익'):
            index = i+1
            break
            
    
    # 테이블의 종목당 정보들 
    stocks = bs.find(class_='type_2').find('tbody').find_all('tr')

    result = []

    count = 0
    # count가 주식 수를 채울 때 까지 반복
    for i in range(len(stocks)):
        
        # 테이블에서 종목당 종목 정보만 가져오기
        stock = None
        try:
            no = stocks[i].find(class_="no").get_text()
            stock=stocks[i]
        except AttributeError:
            continue
        
        # PER|ROE|영업이익 N/A가 아니면 종목명 리스트에 넣기
        if(stock.select('td:nth-child({})'.format(index))[0].get_text() != 'N/A'):
            result.append(stock.find(class_="tltle").get_text()) 
            count += 1
        
        # 원하는 종목 만큼 (stockNum == 20)
        if(count>=stockNum):
            break

    driver.quit()

    return result

import json
import os,re
import sys
import urllib.request
from openpyxl import Workbook
from datetime import datetime, timedelta, date

# 네이버 api id password
NEE_CLIENTID = "Y8J4YHW3zwtFxu4WcFvs"
NEE_CLIENTPW = "CG9dbamsK4"


# 날짜(기간)를 구하는 함수!
def dateList(): 
    global datelist
    datelist = []
    for i in range(7): #7일간의 기사를 가져와야 하기 때문에 날짜 데이터를 리스트에 담습니다.
        datecalc = datetime.today() - timedelta(i)
        checkDate = str(datecalc.strftime("%d %b %Y")) #날짜 포멧 만들기 일 월(영문) 년
        datelist.append(checkDate)


#엑셀로 저장하는 함수! 
def makeexcel(headline):
    today = date.today()
    today = re.sub("-", "", str(today.isoformat()))
    filename = "기사모음.csv"
    
    with open(filename, "w", encoding="utf-8", newline='') as fs:
        wr = csv.writer(fs)
        wr.writerow(["종목이름","기사제목"])
        for line in headline:
            wr.writerow(line)
            
    return print("엑셀저장 완료!")
   

# 헤드라인 가져오기 
def getHeadline(stock):
    titlelist=[]
    
    header = {
        "X-Naver-Client-Id" : NEE_CLIENTID,
        "X-Naver-Client-Secret" : NEE_CLIENTPW
    }
    encText = urllib.parse.quote(stock) 

    for i in range(0,1000,100): 
        if len(titlelist) >= 100: break 
        count = 1 + i
        params = "?query=" + encText + "&sort=sim&start="+ str(count) +"&display=100&" #관련도순, count부터 count+100개 가져옵니다
        url = "https://openapi.naver.com/v1/search/news.json"+params 

        request = urllib.request.Request(url, headers=header)
        response =urllib.request.urlopen(request)
        rescode = response.getcode()
        

        if(rescode==200):
            response_body = response.read()
            results = json.loads(response_body.decode('utf-8'))
            
            items = results["items"] # itmes는 딕셔너리를 값으로 가진 리스트입니다.
            
            for dicts in items: #딕셔너리를 하나씩 꺼냅니다.
                innerList = []
                innerList.append(stock)
                titles = filter(dicts) # 뉴스 헤드라인을 다듬습니다.
                if titles != "" : #비어있는것이 아니라면 리스트에 담습니다. 
                    if len(titlelist) >= 100 : break # 100개를 넘었다면 끝냅니다. 
                    innerList.append(titles) 
                    titlelist.append(innerList)
                else :
                    pass            
        else:
            print("Error Code:" + str(rescode))
            
    return titlelist

# 날짜 확인, 필요없는 문자 제거하는 함수 
def filter(items):
    tt = ""
    for day in datelist:
        if day in items["pubDate"]: #pubDate는 게시된 날짜가 들어있습니다. 날짜값을 조회힙니다.
            #key 이름이 title인 value에 헤드라인이 들어있습니다. 
            tt = items["title"].replace("<b>","") # 필요없는 문자를 제거합니다(1)
            tt = tt.replace("</b>","") # 필요없는 문자를 제거합니다(2)
            tt = re.sub("quot;","",tt) # 필요없는 문자를 제거합니다(3)
            tt = re.sub("amp;","",tt) # 필요없는 문자를 제거합니다(4)
            break
        else: pass

    return tt 


def start(stocks):
    dateList() # 날짜리스트 구하기 
    headlines = [] # headlines 안에 리스트로 ["종목이름","뉴스헤드라인"] 값이 들어갑니다. 
    for stk in stocks:
        headlines += getHeadline(stk)

    resultword = makeexcel(headlines)
    return resultword




# 주식 거래량 상위 20위 
stocksList = getStocks()

#뉴스 헤드라인 크롤링 > 엑셀파일로 저장 
start(stocksList)


file = open('기사모음.csv', encoding='utf-8')

type(file)

csvreader = csv.reader(file)

rows = []
names = []
for row in csvreader:
  if row[0] != '종목이름' and row[1]!= '기사제목':
    names.append(row[0])
    rows.append(row[1])


namelist = []
for v in names:
    if v not in namelist:
        namelist.append(v)

print(namelist)

for s in rows:
  print(getSentimentValue(s, tok, max_len, batch_size, device),s)

counts=[0 for i in range(20)]
newscounts=[0 for i in range(20)]

for c, n in enumerate(namelist):    
    for i, s in enumerate(rows):
        if n == names[i]:
            newscounts[c] = newscounts[c] + 1
            if getSentimentValue(s, tok, max_len, batch_size, device)[0] == 1:
                counts[c] = counts[c] + 1

print(counts, newscounts)


percentAverage = 0

for i in range(len(counts)):
    percentAverage += counts[i]/newscounts[i]

percentAverage /= 20


recommendations = []
percentage = []


for i in range(len(counts)):
    if (counts[i]/newscounts[i] > percentAverage and newscounts[i] > 50):
        recommendations.append(namelist[i])
        percentage.append(int(counts[i]/newscounts[i]*100))

print(recommendations, percentage)
print(namelist)

dictionary0 = dict(zip(percentage, recommendations))
dictionary0 = sorted(dictionary0.items(), reverse=True)

print(dictionary0)