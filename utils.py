import warnings
warnings.filterwarnings('ignore')
import os
from matplotlib import pyplot as plt
import sys
sys.path.append('./detection/darkflow')
from darkflow.net.build import TFNet


def get_detection_model(model_path, model_name='yolo_v2'):
    MODEL_PATH = os.path.join(model_path, 'model', model_name)
    options = {'model': os.path.join(MODEL_PATH, '{}.cfg'.format(model_name)),
               'labels': os.path.join(MODEL_PATH, 'labels.txt'),
               'backup': MODEL_PATH,
               'load': 50500,
               'threshold': 0.5,
               'gpu': 1.0}
    tfnet = TFNet(options)
    return tfnet


def plot_rectangle(bbox, ax, class_name, edgecolor, confidence=None):
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]
    left = xmin
    right = xmax
    top = ymin
    bot = ymax
    ax.add_patch(
        plt.Rectangle((left, top),
                      right - left,
                      bot - top, fill=False,
                      edgecolor=edgecolor, linewidth=3.5)
    )
    label = '{:s}'.format(class_name)
    label_pos_y = top - 10
    if confidence:
        label += ' {0:.2f}'.format(confidence)
        label_pos_y = bot + 20
    ax.text(left, label_pos_y, label,
            bbox=dict(facecolor=edgecolor, alpha=0.5),
            fontsize=14, color='white')
