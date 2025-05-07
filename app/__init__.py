from flask import Flask, jsonify
from flask_restx import Api
from flask_cors import CORS
from flask.json.provider import DefaultJSONProvider
import json
from datetime import datetime


class CustomJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        kwargs.update({
            'ensure_ascii': False,  
            'indent': 2,
            'sort_keys': True,
            'default': self.default,
        })
        return json.dumps(obj, **kwargs)

    def default(self, o):
        if isinstance(o, (datetime.date, datetime.datetime)):
            return o.isoformat()
        return super().default(o)


def create_app():
    app = Flask(__name__)
    
    app.config.update({
        'JSON_AS_ASCII': False,  
        'JSON_SORT_KEYS': False,  
        'JSONIFY_PRETTYPRINT_REGULAR': True,
        'JSONIFY_MIMETYPE': 'application/json; charset=utf-8'
    })
    
    app.json = CustomJSONProvider(app) 
    app.config['JSON_AS_ASCII'] = False  

    CORS(app)
    
    api = Api(
        app,
        title='WebPage Analyzer',
        version='1.0',
        doc='/api/v1',
        default='Документация API',
        default_label='Документация API'
    )
    
    from .schemas import register_models
    register_models(api)
    
    from .routes import api as analyzer_api
    api.add_namespace(analyzer_api)
    
    return app