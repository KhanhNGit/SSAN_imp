import math
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import onnxruntime as ort

from retina_face import *
from configs import parse_args_pred

app = FaceAnalysis()
app.prepare()

def crop_face_from_scene(image, box, scale=1.3):
    y1,x1,w,h=box
    y2=y1+w
    x2=x1+h
    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=np.int16(max(math.floor(y1),0))
    x1=np.int16(max(math.floor(x1),0))
    y2=np.int16(min(math.floor(y2),w_img))
    x2=np.int16(min(math.floor(x2),h_img))
    region=image[x1:x2,y1:y2]
    return Image.fromarray(region[:,:,::-1])

def crop_face(img_path):
    try:
        img = cv2.imread(img_path)
        faces = app.get(img)
        box = faces[0][0], faces[0][1], faces[0][2]-faces[0][0], faces[0][3]-faces[0][1]
        if box[2] < 30 or box[3] < 30:
            return [False, 0]
        return [True, crop_face_from_scene(image=img, box=box)]
    except:
        return [False, 0]

def transform(image):
    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transformed_image = img_transforms(image)
    return transformed_image

def softmax(x):
    exp_x = np.exp(x)
    exp_x = exp_x / exp_x.sum(axis=0)
    exp_x = np.round(exp_x*100, 2)
    return exp_x

model_path = './weights/best_model.onnx'
sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_names = [i.name for i in sess.get_inputs()]
input_shape = sess.get_inputs()[0].shape

def main(args):
    img_path = args.image_name
    if crop_face(img_path)[0] == False:
        print("The input image is corrupted. Please try another image")
    else:
        image = crop_face(img_path)[1]
        transformed_image = transform(image)
        input_tensor = transformed_image.reshape(input_shape).cpu().numpy()
        outputs = sess.run(None, {input_names[0]: input_tensor, input_names[1]: input_tensor})
        outputs = softmax(outputs[0][0])
        if outputs[0]>=outputs[1]:
            print('Spoof\n', 'Confident: {}%'.format(outputs[0]))
        else:
            print('Live\n', 'Confident: {}%'.format(outputs[1]))

if __name__ == '__main__':
    args = parse_args_pred()
    main(args=args)
