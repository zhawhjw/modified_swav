import sys,os
import numpy as np
from sklearn.manifold import TSNE
import cv2
from matplotlib import pyplot as plt
import json
import re, os
from PIL import Image
from functools import reduce 

def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center,
    # compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y



model_info_json = "C:/Users/tomer/Datasets/model_info.json"
query_to_reqs = {}
query_to_reqs_w_info = {}
model_info = {}

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


def load_results_file(path_to_results_file):
    global query_to_reqs, query_to_reqs_w_info, model_info
    with open(path_to_results_file,'r') as f:
        next(f)
        for line in f:
            linearr = line.strip().split(",")
            query = linearr[0]
            query_relative_path = os.path.relpath(query, data_path)
            img_id = query_relative_path.split(os.sep)[-1].split(".")[0]
            print(img_id,query_relative_path)
            exit()
            values = linearr[1:]
            image_files = values[::2]
            embeddings = values[1::2]
            if img_id in model_info:
                values = model_info[img_id]
                model_info[query_relative_path] = values
                #print(model_info[query_relative_path])
            value_imgs_relative_path = [os.path.relpath(img, data_path) for img in image_files]
            query_to_reqs[query_relative_path] = value_imgs_relative_path
            query_to_reqs_w_info[query_relative_path] = [{"embedding":embedding, "img":img  } for embedding,img in zip(embeddings,value_imgs_relative_path)] 


load_model_info(model_info_json)
#path_to_results_file = """C:/Users/tomer/researchsoftware/styleestimation/results_log_similar_images.csv"""
#load_results_file(path_to_results_file)

def getImage(path):
    return OffsetImage(plt.imread(path),zoom=0.041)

def get_label_elements_of_style(idx):
    pass


def get_discrete_position_old(xs, ys, tile_dim):
    min_xs, max_xs = min(xs), max(xs)
    min_ys, max_ys = min(ys), max(ys)
    width = max_xs - min_xs 
    height = max_ys - min_ys
    cell_size = 1.7
    cells = {}
    ctr = 0
    for x,y in zip(xs,ys):
        x_pos = int( (x - min_xs)/cell_size ) 
        y_pos = int( (y - min_ys)/cell_size )
        if (x_pos,y_pos) in cells:
            cells[(x_pos,y_pos)]+=[ctr]
        else:
            cells[(x_pos,y_pos)] = [ctr]
        ctr+=1
    elems_in_cell = [ len(cells[cell]) for cell in cells]
    # TODO binary search to find 4 elems in cell 
    # fit to grid, each cell can fit 4 elems 
    print('max elems in cell', max(elems_in_cell))
    if max(elems_in_cell) >4:
        print("error, more elems in cell than allowed")
        exit()
    elem_pos = {}
    for cell in cells:
        elem_list = cells[cell]
        for k,elem in enumerate(elem_list):
            x_pos, y_pos = cell 
            dx = cell_size /2. * int( k % 2 )
            dy = cell_size /2. * int( k / 2 )
            elem_pos[elem] = x_pos + dx,  y_pos + dy 

    X = [elem_pos[key] for key in sorted(elem_pos)] 
    return X 

def get_discrete_position(xs, ys, img_dim, cell_size = 1.3, min_cell_size = 0.01, max_cell_size = 100.0):
    min_xs, max_xs = min(xs), max(xs)
    min_ys, max_ys = min(ys), max(ys)
    width = max_xs - min_xs 
    height = max_ys - min_ys
    cells = {}
    ctr = 0
    for x,y in zip(xs,ys):
        x_pos = int( (x - min_xs)/cell_size ) 
        y_pos = int( (y - min_ys)/cell_size )
        if (x_pos,y_pos) in cells:
            cells[(x_pos,y_pos)]+=[ctr]
        else:
            cells[(x_pos,y_pos)] = [ctr]
        ctr+=1
    elems_in_cell = [ len(cells[cell]) for cell in cells]
    # TODO binary search to find 4 elems in cell 
    # fit to grid, each cell can fit 4 elems 
    print('max elems in cell', max(elems_in_cell),cell_size,min_cell_size,max_cell_size )
    tile_dim = 2*img_dim 
    # Binary Search on elems in cell to minimize distances 
    if max(elems_in_cell) >2:
        #print("error, more elems in cell than allowed")
        return get_discrete_position(xs, ys, img_dim, (cell_size + min_cell_size)/2.0 , min_cell_size, cell_size)
    elif max(elems_in_cell) < 1.7:
        return get_discrete_position(xs, ys, img_dim, (cell_size + max_cell_size)/2.0 , cell_size, max_cell_size)
    elem_pos = {}
    for cell in cells:
        elem_list = cells[cell]      
        elems = [ (xs[elem],ys[elem])   for elem in elem_list]
        elems = sorted(elems, key = lambda t: t[0])
        for k,elem in enumerate(elem_list):
            x_pos, y_pos = cell 
            dx = int( tile_dim /2. * int( k % 2 ))
            dy = int( tile_dim /2. * int( k / 2 ) ) 
            elem_pos[elem] = tile_dim* x_pos + dx,  tile_dim* y_pos + dy 

    X = [elem_pos[key] for key in sorted(elem_pos)] 
    return X 



npy_file_ = 'main_swav_elements_of_style_embeds.npy'
csv_file_ = 'main_swav_elements_of_style_files.npy'


prefix = npy_file_.split(".")[0]
save_folder = "visualization"

IMG_DIM = 32 # must divide by 4!!
assert IMG_DIM % 4 == 0

with open(npy_file_, 'rb') as f:
    embeddings = np.load(f)
    nsamples = embeddings.shape[0]
    dims = reduce( lambda x, y:x*y  , embeddings.shape[1:]  )
    embeddings_reshaped = embeddings.reshape((nsamples,dims))
    tsne = TSNE(n_components=2,perplexity=35.0, random_state=42 ).fit_transform(embeddings_reshaped)

    xs = np.array( [e[0] for e in tsne] ) 
    ys = np.array( [e[1] for e in tsne] )
    xs_org = np.copy(xs)
    ys_org = np.copy(ys)


    with open(csv_file_, 'rb') as fi:        
        image_paths = np.load(fi)

        for idx1, embedding1 in enumerate(embeddings):
            image = image_paths[idx1]
            img_id = image.split(os.sep)[-1].split(".")[-2]
            #supercategory = model_info[img_id]['supercategory']

        if True:
            paths = [image_paths[idx1] for idx1, embedding1 in enumerate(embeddings)]

            X = get_discrete_position(xs, ys, IMG_DIM)
            xs = [e[0] for e in X if e[0] ]
            ys = [e[1] for e in X if e[1] ] 
            print('max x and y',max(xs),max(ys))
            print('len xs and ys', len(xs), len(ys))

            if False:
                start_x, start_y  = 0, 0
                end_x, end_y = 0,0
                xs = [e[0] for e in X if e[0] if e[0] < 5000 ]
                ys = [e[1] for e in X if e[1] if e[1] < 5000 ] 
                
                if False:
                    with open("embeddings_06022021.txt",'w') as f_example:
                        f_example.write("file,x,y,width_and_height\n")
                        for x0, y0, path in zip(xs, ys,paths):
                            path_ = (os.sep).join( path.split(os.sep)[-2:])  
                            str_ = "%s,%d,%d,%d\n" % ( str(path_), x0,y0, IMG_DIM) 
                            f_example.write(str_)

                img = Image.new('RGB', (max(xs) + IMG_DIM, max(ys) + IMG_DIM),"WHITE" )
                for x0, y0, path in zip(xs, ys,paths):
                    image = Image.open(path).convert('L')
                    new_image = image.resize((IMG_DIM, IMG_DIM))
                    img.paste(new_image, (x0, y0))
                    print(x0,y0)
                save_path = save_folder + os.sep + prefix + "_images.png"
                img.save(save_path)

        if False: # here we show the style distribution
            styles =  [  model_info[( image.split(os.sep)[-1].split(".")[-2] )]['supercategory']
                              for image in image_paths   ]
            style_list = list(set(styles)) # works with 3 styles
            # https://xkcd.com/color/rgb/
            # b= [ x[1:-1] for x in a.split("\n") if len(x.strip())>0 and '#' in x ] 
            colors = ['#8e82fe', '#53fca1', '#aaff32', '#380282', '#ceb301', '#ffd1df', 
            '#cf6275', '#0165fc', '#0cff0c', '#c04e01', '#04d8b2', '#01153e', '#3f9b0b', 
            '#d0fefe', '#840000', '#be03fd', '#c0fb2d', '#a2cffe', '#dbb40c', '#8fff9f', 
            '#580f41', '#4b006e', '#8f1402', '#014d4e', '#610023', '#aaa662', '#137e6d', '#7af9ab', '#02ab2e']
            style_colors = [colors[style_list.index(style)] for style in styles]
            plt.figure(figsize=(6, 5))
            for x,y,c in zip(xs_org,ys_org,style_colors):
                plt.scatter(x, y, color=c)
            plt.legend()
            #plt.show()
            category_file = save_folder + os.sep + prefix + "_supercategories.png"
            plt.savefig(category_file)

        # TODO - show class/super category distribution 
        # TODO - show images instead of scatter plot 
        if False: # here we show the style distribution
            styles = [ path.split(os.sep)[9:10][0] for path in image_paths ]
            style_list = list(set(styles)) # works with 3 styles
            # https://xkcd.com/color/rgb/
            # b= [ x[1:-1] for x in a.split("\n") if len(x.strip())>0 and '#' in x ] 
            colors = ['#8e82fe', '#53fca1', '#aaff32', '#380282', '#ceb301', '#ffd1df', 
            '#cf6275', '#0165fc', '#0cff0c', '#c04e01', '#04d8b2', '#01153e', '#3f9b0b', 
            '#d0fefe', '#840000', '#be03fd', '#c0fb2d', '#a2cffe', '#dbb40c', '#8fff9f', 
            '#580f41', '#4b006e', '#8f1402', '#014d4e', '#610023', '#aaa662', '#137e6d', '#7af9ab', '#02ab2e']
            style_colors = [colors[style_list.index(style)] for style in styles]
            plt.figure(figsize=(6, 5))
            for x,y,c in zip(xs_org,ys_org,style_colors):
                plt.scatter(x, y, color=c)
            plt.legend()
            #plt.show()   
            style_file = save_folder + os.sep + prefix+ "_styles.png"
            plt.savefig(style_file)

        if True: # here we show the style distribution
            styles = [ path.split(os.sep)[-2].split("_")[0] for path in paths]   # shows category for elements of style dataset
            style_list = list(set(styles)) # works with 3 styles
            colors = ['#8e82fe', '#53fca1', '#aaff32', '#380282', '#ceb301', '#ffd1df', 
            '#cf6275', '#0165fc', '#0cff0c', '#c04e01', '#04d8b2', '#01153e', '#3f9b0b', 
            '#d0fefe', '#840000', '#be03fd', '#c0fb2d', '#a2cffe', '#dbb40c', '#8fff9f', 
            '#580f41', '#4b006e', '#8f1402', '#014d4e', '#610023', '#aaa662', '#137e6d', '#7af9ab', '#02ab2e']
            style_colors = [colors[style_list.index(style)] for style in styles]
            plt.figure(figsize=(21, 21))
            for x,y,c,l in zip(xs_org,ys_org,style_colors,styles):
                plt.scatter(x, y, color=c, label=l)

            handles, labels = plt.gca().get_legend_handles_labels()
            newLabels, newHandles = [], []
            for handle, label in zip(handles, labels):
                if label not in newLabels:
                    newLabels.append(label)
                    newHandles.append(handle)
            plt.legend(newHandles, newLabels)
            #plt.legend()
            #plt.show()   
            style_file = save_folder + os.sep + prefix+ "_styles.png"
            plt.savefig(style_file)

        if True: # general scatter plot - need to actually show classes 
            plt.figure(figsize=(19, 19))
            c = 'r' 
            plt.scatter(xs,ys, c=c, label = "atra ")
            plt.legend()
            plt.show()    

        exit()

        print(image_paths)
        for idx1, embedding1 in enumerate(embeddings):
            image = image_paths[idx1] 
            x,y = tsne[idx1][0], tsne[idx1][1]
            print(image,x,y)
            pass

        exit()
        tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

        # now we'll put a small copy of every image to its corresponding T-SNE coordinate
        for image_path, label, x, y in tqdm(
                zip(images, labels, tx, ty),
                desc='Building the T-SNE plot',
                total=len(images)
        ):
            image = cv2.imread(image_path)

            # scale the image to put it to the plot
            image = scale_image(image, max_image_size)

            # draw a rectangle with a color corresponding to the image class
            image = draw_rectangle_by_class(image, label)

            # compute the coordinates of the image on the scaled plot visualization
            tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

            # put the image to its t-SNE coordinates using numpy sub-array indices
            tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

        cv2.imshow('t-SNE', tsne_plot)
        cv2.waitKey()
