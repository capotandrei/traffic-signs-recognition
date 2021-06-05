import warnings
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import time

from utils import *
import cv2


tfnet = get_detection_model(model_name='yolo_v2')

min_score_thresh = 0.5

fig, ax = plt.subplots(figsize=(20, 20))
image_path = '00500.png'
# image_path = '00507.png'
# image_path = '00599.png'
image_path = '00165.png'
image = Image.open(image_path)
image_name = os.path.basename(image_path)
width, height = image.size
ax.imshow(image)

start = time.time()
image_np = np.array(image)
image_np = image_np[:, :, ::-1]  # rgb -> bgr
pred_results = tfnet.return_predict(image_np)
print(time.time() - start)
# print(pred_results)
for idx, det in enumerate(pred_results):
    score = det['confidence']
    if score > min_score_thresh:
        bbox = det['topleft']['x'], det['topleft']['y'], det['bottomright']['x'], det['bottomright']['y']
        plot_rectangle(bbox, ax, det['label'], 'red', score)


        # print(image_np.shape)
        # print()
        # im = np.array(image)
        # img = im[det['topleft']['y']:det['bottomright']['y'],
        #       det['topleft']['x']:det['bottomright']['x']]
        # cv2.imshow(f'image {idx}', img)
        # if cv2.waitKey(0) == ord('q'):
        #     continue
        #
        # print('a trecut de aci')


plt.draw()
fig.tight_layout()
plt.axis('off')
plt.show()
