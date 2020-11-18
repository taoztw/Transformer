"""
Created bt tz on 2020/11/14 
"""

__author__ = 'tz'

import json

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from one_trans import machine_translate
from flask_cors import *
from datetime import datetime

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@127.0.0.1:3306/message?charset=utf8'
CORS(app, supports_credentials=True)
db = SQLAlchemy(app)

class Message(db.Model):

    __tablename__ = 'message'

    id = db.Column(db.Integer, primary_key=True)
    create_time = db.Column(db.Integer)
    info = db.Column(db.String(200))





@app.route('/message',methods=['post'])
def save_message():
    "提交留言功能"
    res = jsonify({
        'meg': 'success',
        'code': 200,
        'result': ''
    })
    res.headers['Access-Control-Allow-Origin'] = '*'
    res.headers['Access-Control-Allow-Methods'] = 'POST，GET,OPTIONS'
    res.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    mes = request.json.get('message')
    message = Message(info=mes, create_time=int(datetime.now().timestamp()))
    db.session.add(message)
    db.session.commit()
    return res


@app.route('/translation',methods=['post'])
def translate():
    "翻译功能"
    result = {}
    if request.method == 'POST':
        data = request.json
        # print(data, type(data))
        sentence = data.get('sentence')
        print(sentence)
        cn_trans = machine_translate(sentence)
        result['result'] = cn_trans
        result['msg'] = 'success'
        result['code'] = 200
    res = jsonify(result)
    res.headers['Access-Control-Allow-Origin'] = '*'
    res.headers['Access-Control-Allow-Methods'] = 'POST，GET,OPTIONS'
    res.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    # print(res)
    return res

@app.route('/list',methods=['get'])
def message_list():
    result = []
    data = Message.query.all()
    for i in data:
        d = {}
        d['message'] = i.info
        d['create_time'] = i.create_time
        result.append(d)
    return jsonify(result)

if __name__ == '__main__':
    # sentence = 'I am a boy'
    # print(cn_idx2word)
    # print(machine_translate(sentence))

    # 开启测试服务端code
    db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)