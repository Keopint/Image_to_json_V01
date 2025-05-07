from flask_restx import Model, fields

element_position_model = Model("ElementPosition", {
    "x": fields.Integer(required=True),
    "y": fields.Integer(required=True),
    "width": fields.Integer(required=True),
    "height": fields.Integer(required=True)
})

web_element_model = Model("WebElement", {
    "type": fields.String(required=True, enum=["text", "button", "image", "input"]),
    "position": fields.Nested(element_position_model),
    "text": fields.String,
    "image_embedding": fields.String(description="Base64 encoded image")  
})

web_page_structure_model = Model("WebPageStructure", {
    "elements": fields.List(fields.Nested(web_element_model)),
    "width": fields.Integer(required=True),
    "height": fields.Integer(required=True)
})

element_difference_model = Model("ElementDifference", {
    "change_type": fields.String(required=True, enum=["added", "removed", "modified", "moved"]),
    "old_position": fields.Nested(element_position_model, allow_null=True),
    "new_position": fields.Nested(element_position_model, allow_null=True),
    "old_text": fields.String(allow_null=True),
    "new_text": fields.String(allow_null=True),
    "similarity_score": fields.Float(description="Степень различия от 0 до 1")
})

page_comparison_model = Model("PageComparison", {
    "differences": fields.List(fields.Nested(element_difference_model)),
    "added_count": fields.Integer,
    "removed_count": fields.Integer,
    "modified_count": fields.Integer,
    "moved_count": fields.Integer
})

def register_models(api):
    api.models[element_position_model.name] = element_position_model
    api.models[web_element_model.name] = web_element_model
    api.models[web_page_structure_model.name] = web_page_structure_model
    api.models[element_difference_model.name] = element_difference_model
    api.models[page_comparison_model.name] = page_comparison_model