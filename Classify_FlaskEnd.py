from flask import Flask, request, jsonify
import re
import joblib
import pymorphy2
import os

app = Flask(__name__)
# 加载分类模型
script_dir = os.path.dirname(__file__)
classifier_path = os.path.join(script_dir, "Multi_label_classifier.pkl")
vectorizer_path = os.path.join(script_dir, "Multi_label_vectorizer.pkl")
classifier = joblib.load(classifier_path)
# 加载编码器
vectorizer = joblib.load(vectorizer_path)


def classify_words(text):
    mid_list = []
    high_list = []
    pro4_list = []
    pro8_list = []
    # 对字符串文本进行分词和找原型
    # 加载pymorphy2的俄语处理器
    morph = pymorphy2.MorphAnalyzer()
    words_sentence = re.findall(r'\b\w+\b', text)  # 表示的是句子拆分后的词
    words_origins = []
    for word in words_sentence:
        parsed_word = morph.parse(word)
        if parsed_word:
            words_origins.append(parsed_word[0].normal_form)
            # print(f"Word: {word}, Lemma: {parsed_word[0].normal_form}")
    # 对文本单词进行编码,用的transform,训练用fit_transform
    X = vectorizer.transform(words_origins)
    # 对文本单词进行分类
    y_pred = classifier.predict(X)
    # 根据分类结果，组合返回的四个词汇列表
    for i, pred_labels in enumerate(y_pred):
        if pred_labels[0] == 1:
            pro4_list.append(words_sentence[i])
        if pred_labels[1] == 1:
            pro8_list.append(words_sentence[i])
        if pred_labels[2] == 1:
            high_list.append(words_sentence[i])
        if pred_labels[3] == 1:
            mid_list.append(words_sentence[i])
        # 当pre_labels为[0,0,0,0]时，归类到专八词语
        if all(label == 0 for label in pred_labels):
            pro8_list.append(words_sentence[i])

    all_pro8_words = set(pro8_list + pro4_list + high_list + mid_list)
    all_pro4_words = set(pro4_list + high_list + mid_list)
    all_high_words = set(high_list + mid_list)

    # 专八词汇
    pro8_list = list(all_pro8_words)

    # 专四词汇
    pro4_list = list(all_pro4_words)

    # 高中词汇
    high_list = list(all_high_words)


    out_mid_list = list(set(pro4_list + pro8_list + high_list) - set(mid_list))
    out_high_list = list(set(pro4_list + pro8_list + mid_list) - set(high_list))
    out_pro4_list = list(set(pro8_list + high_list + mid_list) - set(pro4_list))
    out_pro8_list = list(set(mid_list + high_list + pro4_list) - set(pro8_list))
    pro4_list = list(set(pro4_list))
    pro8_list = list(set(pro8_list))
    mid_list = list(set(mid_list))
    high_list = list(set(high_list))
    mid_list_len = len(mid_list)
    high_list_len = len(high_list)
    pro4_list_len = len(pro4_list)
    pro8_list_len = len(pro8_list)


    return {"mid_list": mid_list, "high_list": high_list, "pro4_list": pro4_list, "pro8_list": pro8_list,
            "out_mid_list": out_mid_list, "out_high_list": out_high_list, "out_pro4_list": out_pro4_list,
            "out_pro8_list": out_pro8_list,"mid_list_len":mid_list_len,"high_list_len":high_list_len,
            "pro4_list_len":pro4_list_len,"pro8_list_len":pro8_list_len
            }


# 这里放你分类的逻辑
# 返回分类结果，格式可以是一个字典，如 {"pro4": [...], "pro8": [...], "hig": [...], "mid": [...]}

@app.route('/getlabel', methods=['Post', 'Get'])
def index():
    if request.method == 'POST':
        try:
            text = request.json
            text = text['text']
            results = classify_words(text)
            # return {"message":"success","data":results}
            return jsonify({"message": "success", "data": results}), 200
        except Exception as e:
            # return {"message":"error","data":str(e)}
            return jsonify({"message": "error", "data": str(e)}), 500
    return jsonify({"message": "error", "data": "Invalid Request! Please use post request!"}), 405


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
