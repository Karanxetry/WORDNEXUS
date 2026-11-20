from tensorflow.keras.models import load_model
model=load_model('word_predictor_lstm.h5') 

print(model.input_shape)