from werkzeug.datastructures import FileStorage  
from flask_restx import Namespace, Resource, fields
from flask import request
import werkzeug
import numpy as np
import cv2
from .services import analyze_webpage_image, compare_pages
from .schemas import web_page_structure_model, page_comparison_model

api = Namespace('analyzer', description='Анализ веб-страниц')


upload_parser = api.parser()
upload_parser.add_argument(
    'image', 
    type=werkzeug.datastructures.FileStorage,
    location='files',
    required=True,
    help='PNG/JPG изображение'
)


compare_parser = api.parser()
compare_parser.add_argument(
    'image1', 
    type=FileStorage,  
    location='files',
    required=True,
    help='Первая версия страницы (PNG/JPG)'
)
compare_parser.add_argument(
    'image2',
    type=FileStorage,
    location='files',
    required=True,
    help='Вторая версия страницы (PNG/JPG)'
)

@api.route('/compare')
class ComparePages(Resource):
    @api.expect(compare_parser)
    @api.marshal_with(page_comparison_model)
    def post(self):
        args = compare_parser.parse_args()
        
        try:
            # Обработка первого изображения
            img1_bytes = args['image1'].read()
            img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            # Обработка второго изображения
            img2_bytes = args['image2'].read()
            img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if img1 is None or img2 is None:
                api.abort(400, "Некорректный формат изображения")
                
            return compare_pages(img1, img2)
            
        except Exception as e:
            api.abort(500, f"Ошибка при сравнении: {str(e)}")


@api.route('/analyze')
class AnalyzeImage(Resource):
    @api.expect(upload_parser)
    @api.marshal_with(web_page_structure_model)
    def post(self):
        args = upload_parser.parse_args()
        image_file = args['image']
        
        try:
            # Чтение изображения
            img_bytes = image_file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                api.abort(400, "Invalid image format")
                
            return analyze_webpage_image(img)
            
        except ValueError as e:
            api.abort(422, str(e))
        except Exception as e:
            api.abort(500, f"Server error: {str(e)}")