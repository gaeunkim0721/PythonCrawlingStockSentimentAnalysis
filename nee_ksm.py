from fileinput import filename
import requests
import json
from datetime import date
from 파이썬팀과제 import dictionary0


#인가코드 받기 
# REST API : '772e5236d7a793ed613505af7b6beb15'
#https://kauth.kakao.com/oauth/authorize?response_type=code&client_id=772e5236d7a793ed613505af7b6beb15&redirect_uri=https://example.com/oauth



url = 'https://kauth.kakao.com/oauth/token'
rest_api_key = '772e5236d7a793ed613505af7b6beb15'
filename = "./kakao_code.json"

#최초 세팅.. token_refresh 안될때 
def set_tokens():
    redirect_uri = 'https://example.com/oauth'
    authorize_code = 'Z7S4d-N7vmdBnv0wdYzIxxaWRB3OnogKPXwkq7Xt9RAXXZnSI-iKjxCNjWYpeVrB0XA4Xgopb1UAAAGBjfQIsA'

    data = {
        'grant_type':'authorization_code',
        'client_id':rest_api_key,
        'redirect_uri':redirect_uri,
        'code': authorize_code,
    }

    response = requests.post(url, data=data)
    tokens = response.json()
    save_tokens(tokens)
    print(tokens)

    return tokens


def save_tokens(tokens):
    with open(filename, "w") as fp:
        json.dump(tokens, fp)

# 읽어오는 함수
def load_tokens():
    with open(filename) as fp:
        tokens = json.load(fp)
    return tokens


def token_refresh():
    loadtk = load_tokens()
    
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": rest_api_key,
        "refresh_token": loadtk["refresh_token"]
    }
    
    response = requests.post(url, data=data)
    
    loadtk['access_token'] = response.json()['access_token']
    save_tokens(loadtk)
    
    return loadtk

    
    

    

nee_template_id = 78646
date = str(date.today().strftime("%Y년 %m월 %d일")) 

def sendmsg(token):

    url="https://kapi.kakao.com/v2/api/talk/memo/send"

    # kapi.kakao.com/v2/api/talk/memo/send

    headers={
        "Authorization" : "Bearer " + token["access_token"]
    }

    data={
        "template_id" : nee_template_id,
        "template_args" : json.dumps({
        "${TOP1_H}" : "삼성전자",
        "${TOP1_B}" : "긍정",
        "${TOP2_H}" : "현대자동차",
        "${TOP2_B}" : "긍정",
        "${TOP3_H}" : "카카오",
        "${TOP3_B}" : "긍정",
        "${TOP4_H}" : "신한지주",
        "${TOP4_B}" : "긍정",
        "${TOP5_H}" : "고려아연",
        "${TOP5_B}" : "긍정",
        "${TODAY}" : date,
        }),
    }

    response = requests.post(url, headers=headers, data=data)
    print(response.status_code)
    print(response.text)


sendmsg(token_refresh())