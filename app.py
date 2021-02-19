from flask import Flask, request, jsonify
from classify import do_predict

# flask run --port=8765
server = Flask(__name__)


def response(result, status='0'):
    return jsonify({'result': result, 'status': status, })


@server.route('/predict')
def predict():
    content = request.args.get('content', '', type=str)
    if '' == content:
        return response('文本不能为空', status='-1')
    return response(do_predict(content))
