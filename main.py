import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pickle
import streamlit as st

def toxic_prediction(comment):
    model = tf.keras.models.load_model('toxic_text_classification.h5')
    from_disk = pickle.load(open('tv_layer.pkl', "rb"))
    vectorizer = TextVectorization(max_tokens=from_disk['config']['max_tokens'],
                                            output_mode=from_disk['config']['output_mode'],
                                            output_sequence_length=from_disk['config']['output_sequence_length'])
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    vectorizer.set_weights(from_disk['weights'])
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    results = model.predict(vectorized_comment)
    level = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    text = "The Text '{}' is ".format(comment)
    flag = True
    for idx, col in enumerate(level):
        if results[0][idx]>0.5:
            flag = False
            text +='{},\n'.format(col)
        # text +='{}: {}\n'.format(col, results[0][idx]>0.5)
    if flag:
        text += 'Non-toxic '
        
    return text
    # result = []
    # toxic_level = ['Toxic','Severe_toxic','Obscene','Threat','Insult','Identity_hate']
    # for idx,res in enumerate(results[0]):
    #     if res>0.5:
    #         result.append(toxic_level[idx])
    # if not result:
    #     result.append('Non-toxic')
    # return result 

def main():
    
    st.title("Toxic Text Detection")
    
    comment = st.text_input("Enter the text here")
    result = ''
    
    if st.button('Result'):
        result = toxic_prediction(comment)
    
    st.success(result)
    
if __name__ == '__main__':
    main()
    
