from flask import Flask, render_template, request, redirect, url_for
from flask import request
from flask import jsonify
import json
import re, os

STATIC = 'static'
app = Flask(__name__, static_url_path='/' + STATIC)

data_path = os.getcwd() + os.sep + STATIC
print(data_path)

model_info_json = "C:/Users/tomer/Datasets/model_info.json"
query_to_reqs = {}
model_info = {}

def load_results_file(path_to_results_file):
    global query_to_reqs, model_info
    with open(path_to_results_file,'r') as f:
        next(f)
        for line in f:
            linearr = line.strip().split(",")
            query = linearr[0]
            query_relative_path = os.path.relpath(query, data_path)
            img_id = query_relative_path.split(os.sep)[-1].split(".")[0]
            if img_id in model_info:
                values = model_info[img_id]
                model_info[query_relative_path] = values
                #print(model_info[query_relative_path])
            values = linearr[1:]
            image_files = values[::2]
            value_imgs_relative_path = [os.path.relpath(img, data_path) for img in image_files]
            query_to_reqs[query_relative_path] = value_imgs_relative_path

def load_results_with_additional_data(path_to_results_file):
    global query_to_reqs
    def get_style(path_to_img):
        query_style_w_path = os.path.dirname(query_relative_path)
        query_style = query_style_w_path.split(os.sep)[-1]
        return query_style
    with open(path_to_results_file,'r') as f:
        next(f)
        for line in f:
            linearr = line.strip().split(",")
            query = linearr[0]
            query_relative_path = os.path.relpath(query, data_path)
            values = linearr[1:]
            image_files = values[::2]
            value_imgs_relative_path = [os.path.relpath(img, data_path) for img in image_files]
            query_to_reqs[query_relative_path] = value_imgs_relative_path
            print('style',get_style(query_relative_path))
            print(value_imgs_relative_path)
            print(os.path.dirname(value_imgs_relative_path[0]))
            break 

@app.route('/imagegrid', methods=['GET', 'POST'])
def modelqa():
    global query_to_reqs, model_info
    print(model_info)
    return render_template('imagegrid.html', data=query_to_reqs, model_info=model_info)


def load_model_info(path_to_json):
    global model_info
    model_info = {}
    with open(path_to_json,'r') as f:
        data = json.load(f)
        for e in data:
            model_id = e['model_id']
            supercategory = e['super-category']
            category = e['category']
            style = e['style']
            theme = e['theme']
            material = e['material']
            model_info[model_id] = {'supercategory': supercategory, 'category':category, 'style':style, 'theme':theme, 'material':material}

if __name__ == "__main__":
    load_model_info(model_info_json)
    path_to_results_file = """C:/Users/tomer/researchsoftware/styleestimation/results_log_similar_images.csv"""
    load_results_file(path_to_results_file)
    #load_results_with_additional_data(path_to_results_file)
    #print("finished init_nodes_by_category")
    #print(  list(query_to_reqs.keys())[5]  )
    app.run(debug=True)
