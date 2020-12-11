import os
import pathlib
import pickle
import warnings
from pathlib import Path

import cupy as cp
import numpy as np
import plotly.graph_objects as go
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

warnings.filterwarnings('ignore')


def load_dataset(container_path, dimension=(64, 64), as_gray=False):
    categories = [f.name for f in os.scandir(container_path) if f.is_dir()]

    train_img = []
    test_img = []
    train_labels = []
    test_labels = []

    for i, cat in enumerate(categories):
        catpath = '/'.join([container_path, cat])
        img_names = [name.name for name in os.scandir(
            catpath) if name.is_file()]
        img_names = sorted(img_names, reverse=True)
        with tqdm(total=len(img_names), desc=cat) as pbar:
            for j, file in enumerate(img_names):
                img_path = '/'.join([catpath, file])
                img = imread(img_path, as_gray=as_gray)

                img = resize(img, (dimension[0], dimension[1]), preserve_range=False)
                if j > 99:
                    train_img.append(img)
                else:
                    test_img.append(img)
                pbar.update(1)
        print('%s category is loaded' % cat)
        train_labels.extend(cp.full((len(img_names) - 100), i), )
        test_labels.extend(cp.full((100), i), )
        print('*******************')

    Xtr = cp.array(train_img)
    ytr = cp.array(train_labels)

    Xtest = cp.array(test_img)
    ytest = cp.array(test_labels)
    return Xtr, ytr, Xtest, ytest, categories

def make_dirs(*dirs):
    for dir in dirs:
        try:
            os.makedirs(dir)
        except:
            print('%s folder already exists'%dir)

def save_model(obj, filename, dir):
    with open(dir+filename+'.pickle', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL) 

def load_model(filename, dir):
    model = None
    with open(dir+filename+'.pickle', 'rb') as output:  # Overwrites any existing file.
        model = pickle.load(output)
    return model


def print_scores(CRRns, ACCR, categories):
    for iclass, crrn in enumerate(CRRns):
        print('CRRn of class %s = %.2f %%' % (categories[iclass], crrn*100.0))

    print('ACCR = %.2f %%' % (ACCR*100.0))



def save_arrays(array_dir, **arrays):
    for name, array in arrays.items():
        cp.save("%s/%s" % (array_dir, name), array)


def plot_loss_vs_epochs(loss_train, loss_val, epochs, title):

    fig = go.Figure(data=go.Scatter(
        x=epochs,
        y=loss_train,
        name="Training"
    ))

    fig.add_trace(go.Scatter(
        x=epochs,
        y=loss_val,
        name="Validation"
    ))

    x_best = np.argmin(loss_val)
    y_best = loss_val[x_best]
    fig = add_arrow_annotation(
        fig, x_best, y_best, text='Stop here: Overfitting!')

    fig.add_shape(type="line",
                  x0=x_best, x1=x_best,
                  line=dict(color="slategray", width=2))

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        yaxis_title="Loss",
        xaxis_title="Epochs"
    )
    # fig.update_xaxes(type="category")
    #fig.write_html('%s/%s_interactive.html' % (plot_dir, fname))
    fig.show()


def plot_tuning_results(params, val_accrs, epochs, param_name):

    fig = go.Figure()

    for i, v in enumerate(params):
        fig.add_trace(go.Scatter(
            x=epochs,
            y=cp.asnumpy(val_accrs[i]),
            name=param_name+" = "+str(v),
        ))

    fig.update_layout(
        title_text=param_name + " Fine-tuning",
        title_x=0.5,
        yaxis_title="Validation Loss",
        xaxis_title='Epochs'
    )
    # fig.update_xaxes(type="category")
    fig.show()


def add_arrow_annotation(fig, x, y, text=''):
    fig.add_annotation(
        x=x,
        y=y,
        xref="x",
        yref="y",
        text=text,
        showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
        ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#000000",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#008000",
        opacity=0.8
    )
    return fig
