from flask import Flask
from nn_service.controllers import api


def create_app():
    appli = Flask(__name__)
    appli.config['CORS_HEADERS'] = 'Content-Type'
    appli.register_blueprint(api, url_prefix='/api/v1/classification')
    return appli