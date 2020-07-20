from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
import os
import nn_service.helper as helper


api = Blueprint('api', __name__)
model = None
class_to_numb = None,
numb_to_class = None

@api.route('/getparts/<string:lang>', methods=['POST'])
@cross_origin()   # @cross_origin(origin='http://localhost', headers=['Content-Type', 'Authorization'])
def classifier(lang):
    """Creates an endpoint that takes in a sentence in json format
    and returns the class according to a loaded model.
    The input JSON format should be:
    {
        "text":"<sentence>"
    }
    the output JSON format is:
    {
         ---> TODO: fix later
    }
    if the response is not json - return an error"""
    if not request.is_json:
        response = jsonify(
            {"message": "ERROR. The mime type needs to be application/json"})
        response.status_code = 415
        return response

    content, model_type, _ = preprocess_input(lang)

    response = helper.classify(content, lang)
    return jsonify(response)


@api.route('/getclasses/<string:lang>', methods=['GET'])
@cross_origin()
def get_classes(lang):
    print('classes hit')
    # content, model_param = preprocess_input(lang)
    model, class_to_numb, numb_to_class = helper.load_model()  # # EmailPartDetectorLinearModel.load(filepath)
    all_classes = [i for i in numb_to_class.values()]
    response = {"message": all_classes}  # get_classes_for_models(model_type)
    return jsonify(response)


@api.route('/getcontent/<string:lang>', methods=['POST'])
@cross_origin(origin='http://localhost', headers=['Content-Type', 'Authorization'])   # @cross_origin(origin='http://localhost', headers=['Content-Type', 'Authorization'])
def identify_content(lang):
    """Creates an endpoint that takes in a sentence in json format
    and returns the class according to a loaded model.
    The input JSON format should be:
    {
        "text":"<sentence>"
    }
    the output JSON format is:
    {
         ---> TODO: fix later
    }
    if the response is not json - return an error"""
    if not request.is_json:
        response = jsonify(
            {"message": "ERROR. The mime type needs to be application/json"})
        response.status_code = 415
        return response

    content, model_type, use_nn = preprocess_input(lang)
    if type(content) == list:
        content = '\n'.join(content)
    print('content', content, '\nuse_nn', use_nn)

    response, signature, prev_email = helper.extract_parts(content, use_nn)
    return jsonify({'content': response, 'signature': signature})


def preprocess_input(lang):
    data = request.get_json()
    sentence = data['text']
    if type(sentence) == str:  # sent as string, not array. Make it Array, embedding service expects array.
        sentence = [sentence]
    use_nn = False
    if 'use_nn' in data:
        use_nn = bool(data['use_nn'])
    # return model type
    model_type = "ecommerce_en_US"
    if lang == 'en':
        model_type = "ecommerce_en_US"
    elif lang == 'sv':
        model_type = "ecommerce_sv_SE"
    elif lang == 'no':
        model_type = "ecommerce_no_NO"
    return sentence, model_type, use_nn


def prepare_response(class_names, probabilities, model_type):
    # jsonify for return
    clsfcns = []   # classifications
    # create dictionary from classes and their probabilities
    for clsn, prob in zip(class_names, probabilities):
        dct = dict()
        dct["classification"] = clsn
        dct["probability"] = prob
        clsfcns.append(dct)
    # jsonify the response
    response = jsonify({
        "type": model_type,
        "classifications": clsfcns
    })
    response.status_code = 200
    return response


