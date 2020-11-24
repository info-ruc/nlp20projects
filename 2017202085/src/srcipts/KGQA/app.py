from flask import Flask, render_template, request, jsonify
from neo_db.query_graph import query_coauthor,get_KGQA_answer,query_author_info,query_paper_info
from KGQA.ltp import get_target_array
app = Flask(__name__)

from KGQA.question_classifier import *
from KGQA.question_parser import *


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])

def index(name=None):
    return render_template('index.html', name = name)


@app.route('/search', methods=['GET', 'POST'])
def search():
    return render_template('search.html')


@app.route('/search_author_info', methods=['GET', 'POST'])
def search_author_info():
    return render_template('search_author_info.html')


@app.route('/search_paper_info', methods=['GET', 'POST'])
def search_paper_info():
    return render_template('search_paper_info.html')


@app.route('/KGQA', methods=['GET', 'POST'])
def KGQA():
    return render_template('KGQA.html')


@app.route('/get_profile',methods=['GET','POST'])
def get_profile():
    name = request.args.get('character_name')
    json_data = get_answer_profile(name)
    return jsonify(json_data)

@app.route('/KGQA_answer', methods=['GET', 'POST'])
def KGQA_answer():
    question = request.args.get('name')
    classifier = QuestionClassifier()
    parser = QuestionPaser()


    res_classify = classifier.classify(question)
    res_sql = parser.parser_main(res_classify)
    print(res_sql)
    json_data = get_KGQA_answer(res_sql)
    print(json_data)
    return jsonify(json_data)


@app.route('/search_name', methods=['GET', 'POST'])
def search_name():
    name = request.args.get('name')
    level = request.args.get('level')
    json_data=query_coauthor(str(name),level)
    print(json_data)
    return jsonify(json_data)


@app.route('/search_name2', methods=['GET', 'POST'])
def search_name2():
    name = request.args.get('name')
    json_data=query_author_info(str(name))
    print(json_data)
    return jsonify(json_data)


@app.route('/search_paper', methods=['GET', 'POST'])
def search_papr():
    name = request.args.get('name')
    json_data=query_paper_info(str(name))
    print(json_data)
    return jsonify(json_data)


@app.route('/get_all_relation', methods=['GET', 'POST'])
def get_all_relation():
    return render_template('all_relation.html')


if __name__ == '__main__':
    app.debug=True
    app.run()
