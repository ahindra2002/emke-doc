import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array,array_to_img
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import tensorflow.keras as K
import boto3
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
from django.core.files.uploadedfile import UploadedFile



session = boto3.Session(
    aws_access_key_id='AKIAVUD24H2B7GK6DBO2',
    aws_secret_access_key='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'
)
s3 = session.client('s3')
def save_img2s3(img,image_name):
    bucket_name = 'mybucket-emke'
    
    url="test"
    #if(isinstance(img, UploadedFile)):
    if(isinstance(img, Image.Image)):
        object_key = f'assets/{image_name}.jpg'
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        s3.upload_fileobj(buffer, bucket_name, object_key)
        url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"
    else:
        object_key = f'assets/{image_name}'
        s3.upload_fileobj(img, bucket_name, object_key)
        url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"
    
    return url

def download_s3model(disease):
    bucket_name = 'mybucket-emke'
    
    object_key="model_weights/"
    local_path='playground/model_weights/'
    if disease=='malaria':
        object_key=object_key+'weights_malaria.h5'
        local_path=local_path+'weights_malaria.h5'
    elif disease=='Brain Cancer':
        object_key=object_key+'weights_brain_cancer.h5'
        local_path=local_path+'weights_brain_cancer.h5'
    elif disease=="pneumonia":
        object_key=object_key+'weights_pneumonia.h5'
        local_path=local_path+'weights_pneumonia.h5'

    s3.download_file(bucket_name,object_key,local_path)
    return local_path
'''
    
def download_froms3(img_url):
    parsed_url = urlparse(img_url)
    object_key = parsed_url.path.lstrip('/')
    prefix = b"assets/"
    index = object_key.find(prefix)

    if index != -1:
        image_key = object_key[index:]
        image_name= object_key[index+len(prefix):]
    s3 = boto3.client('s3')
    bucket_name = 'mybucket-emke'
    image_key=image_key.decode('utf-8')
    image_name=image_name.decode('utf-8')
    s3.download_file(bucket_name, image_key, image_name)

    return image_name

    '''

    


def pwd_strength(password):
    l, u, p, d = 0, 0, 0, 0
    s =password
    capitalalphabets="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    smallalphabets="abcdefghijklmnopqrstuvwxyz"
    specialchar="$@_"
    digits="0123456789"
    if (len(s) >= 8):
        for i in s:
            if (i in smallalphabets):
                l+=1
            if (i in capitalalphabets):
                u+=1
            if (i in digits):
                d+=1
            if(i in specialchar):
                p+=1
    if (l>=1 and u>=1 and p>=1 and d>=1):
        return True
    else:
        return False
    

def VizGradCAM(model, image,labels, interpolant=0.5, plot_results=True):

    """VizGradCAM - Displays GradCAM based on Keras / TensorFlow models
    using the gradients from the last convolutional layer. This function
    should work with all Keras Application listed here:
    https://keras.io/api/applications/
    Parameters:
    model (keras.model): Compiled Model with Weights Loaded
    image: Image to Perform Inference On
    plot_results (boolean): True - Function Plots using PLT
                            False - Returns Heatmap Array
    Returns:
    Heatmap Array?
    """
    #sanity check
    assert (interpolant > 0 and interpolant < 1), "Heatmap Interpolation Must Be Between 0 - 1"

    #STEP 1: Preprocesss image and make prediction using our model
    #input image
    original_img = np.asarray(image, dtype = np.float32)
    #expamd dimension and get batch size
    img = np.expand_dims(original_img, axis=0)
    #predict
    prediction = model.predict(img)
    #prediction index
    print(prediction)
    prediction_idx = np.argmax(prediction)
    label=labels[prediction_idx]
    #print(label)

    #STEP 2: Create new model
    #specify last convolutional layer
    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D))
    target_layer = model.get_layer(last_conv_layer.name)

    #compute gradient of top predicted class
    with tf.GradientTape() as tape:
        #create a model with original model inputs and the last conv_layer as the output
        gradient_model = Model([model.inputs], [target_layer.output, model.output])
        #pass the image through the base model and get the feature map  
        conv2d_out, prediction = gradient_model(img)
        #prediction loss
        loss = prediction[:, prediction_idx]

    #gradient() computes the gradient using operations recorded in context of this tape
    gradients = tape.gradient(loss, conv2d_out)

    #obtain the output from shape [1 x H x W x CHANNEL] -> [H x W x CHANNEL]
    output = conv2d_out[0]

    #obtain depthwise mean
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))


    #create a 7x7 map for aggregation
    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
    #multiply weight for every layer
    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]
    #resize to image size
    activation_map = cv2.resize(activation_map.numpy(), 
                                (original_img.shape[1], 
                                 original_img.shape[0]))
    #ensure no negative number
    activation_map = np.maximum(activation_map, 0)
    #convert class activation map to 0 - 255
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    #rescale and convert the type to int
    activation_map = np.uint8(255 * activation_map)


    #convert to heatmap
    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

    #superimpose heatmap onto image
    original_img = np.uint8((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255)
    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    #heatmap=array_to_img(cvt_heatmap)
    cvt_heatmap = img_to_array(cvt_heatmap)
    heatmap=array_to_img(np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)))

    #enlarge plot
    plt.rcParams["figure.dpi"] = 100

    if plot_results == True:
        fig = plt.figure(figsize=(10, 7))
        rows = 1
        columns = 2
        fig.add_subplot(rows, columns, 1)
        plt.imshow(original_img)
        plt.axis('off')
        plt.title("Image")
        fig.add_subplot(rows, columns, 2)
        plt.imshow(np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)))
        plt.axis('off')
        plt.title("Activation Map")
    
        
    return heatmap,label,tf.get_static_value(prediction[0][prediction_idx])


def predict(img,model_path,resize_value,labels):
  test_model=tf.keras.models.load_model(model_path)
  #img_path=download_froms3(img_path)
  #img = cv2.imread(img_path)
  
  image_array = np.frombuffer(img, np.uint8)
  decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
  img = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)#ertyuijhgf
  img  = cv2.resize(img , (resize_value,resize_value))
  heatmap,label,prediction=VizGradCAM(test_model, img_to_array(img),labels, plot_results=True)
  return heatmap,label,prediction

def predict_malaria(img,resize_value):
    labels={0:'Parasitized',1:'Uninfected'}
    model_path=download_s3model('malaria')
    #model_path='playground/model_weights/weights_malaria.h5'#put in s3
    return predict(img,model_path,resize_value,labels)

def predict_brain_cancer(img,resize_value):
    labels={0:'glioma_tumor',1:'meningioma_tumor', 2: 'no_tumor',3:'pituitary_tumor'}
    model_path=download_s3model('Brain Cancer')
    #model_path='playground/model_weights/weights_brain_cancer.h5'#put in s3
    return predict(img,model_path,resize_value,labels)

def predict_pneumonia(img,resize_value):
    labels={0:'NORMAL',1:'PNEUMONIA'}
    model_path=download_s3model('pneumonia')
    #model_path='playground/model_weights/weights_pneumonia.h5'#put in s3
    return predict(img,model_path,resize_value,labels)


