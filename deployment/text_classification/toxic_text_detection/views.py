import os
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_ocr
from tensorflow.keras.layers import TextVectorization
import pickle



def setup_models():
    path = os.path.join(settings.BASE_DIR,'files','toxic_text_classification.h5') 
    # model = tf.keras.models.load_model('C:/Users/thala/OneDrive/Desktop/Brototype/projects/toxic_text_classification/deployment/text_classification/toxic_text_detection/toxic_text_classification.h5')
    model = tf.keras.models.load_model(path)
    
    path_pickle = os.path.join(settings.BASE_DIR,'files','tv_layer.pkl') 
    from_disk = pickle.load(open(path_pickle, "rb"))
    vectorizer = TextVectorization(max_tokens=from_disk['config']['max_tokens'],
                                            output_mode=from_disk['config']['output_mode'],
                                            output_sequence_length=from_disk['config']['output_sequence_length'])
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    vectorizer.set_weights(from_disk['weights'])
    
    return model,vectorizer

# def score_comment(comment):
#     model,vectorizer = setup_models()
#     vectorized_comment = vectorizer([comment])
#     results = model.predict(vectorized_comment)
#     level = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
#     text = ''
#     for idx, col in enumerate(level):
#         text +='{}: {}\n'.format(col, results[0][idx]>0.5)
#     if text == '':
#         text += 'Non-toxic '
        
#     return text

# def is_toxic(comment):
#     model,vectorizer = setup_models()
#     vectorized_comment = vectorizer([comment])
#     results = model.predict(vectorized_comment)
#     result = {}
#     toxic_level = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
#     for level,res in zip(toxic_level,results[0]):
#         result[level]=res>0.5
#     if not result:
#         result['Non-toxic'] = True
#     return result

def toxic_values(comment):
    model,vectorizer = setup_models()
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    result = []
    toxic_level = ['Toxic','Severe_toxic','Obscene','Threat','Insult','Identity_hate']
    for idx,res in enumerate(results[0]):
        if res>0.5:
            result.append(toxic_level[idx])
    if not result:
        result.append('Non-toxic')
    if result[0] == 'Non-toxic':
        m=1
    else:
        m=0
    return result,m

def ocr_toxic(image):
    pipeline = keras_ocr.pipeline.Pipeline()
    prediction_groups = pipeline.recognize([image])
    # keras_ocr.tools.drawAnnotations(plt.imread(image), prediction_groups[0])
    predicted_image = prediction_groups[0]
    st = ''
    for text, box in predicted_image:
        st += text
        st += ' '
    result,m = toxic_values(st)
    return result,m,st
    
    
def home(request):
   return render(request,'base.html') 
    
def imagefile(request):
    img = request.FILES["img"]
    result,m,input = ocr_toxic(img)
    return render(request,'base.html',{'result':result,'m':m,'input':input})
    
def textfile(request):
    input = request.POST['input_text']
    result,m = toxic_values(input)
    return render(request,'base.html',{'result':result,'m':m,'input':input})
    