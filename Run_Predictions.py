import numpy as np

# API function (return class name)
def return_prediction(model, scaler, json_request):
    s_len = json_request["s_len"]
    s_w = json_request["s_w"]
    p_len = json_request["p_len"]
    p_w = json_request["p_w"]
    
    classes = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    measures =[[s_len, s_w, p_len, p_w]]
    measures_norm = scaler.transform(measures)
    
    flower_class_probabilities = model.predict(measures_norm)
    flower_class_index=np.argmax(flower_class_probabilities,axis=1)
                       
    return classes[flower_class_index]