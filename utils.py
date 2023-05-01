import time
def inf_time_check(model, val_ds):
    images, _ = next(iter(val_ds))
    start_time = time.time()

    # calculate avgerage inference time
    model.predict(images, batch_size=1)
    stop_time1 = time.time()
    model.predict(images, batch_size=1)
    stop_time2 = time.time()
    model.predict(images, batch_size=1)
    stop_time3 = time.time()
    model.predict(images, batch_size=1)
    stop_time4 = time.time()
    model.predict(images, batch_size=1)
    stop_time5 = time.time()

    duration1 = stop_time1 - start_time
    duration2 = stop_time2 - stop_time1
    duration3 = stop_time3 - stop_time2
    duration4 = stop_time4 - stop_time3
    duration5 = stop_time5 - stop_time4
    
    return round((duration1+duration2+duration3+duration4+duration5)*200, 3)


import tensorflow as tf
import tensorflow_addons as tfa
import time
import datetime

from model.vision_transformer import VisionTransformer
from model.vision_performer import VisionPerformer

def ViT_inf_time_check(default_params_1, default_params_2, patch_sizes, test_sample):
    # initialize the model
    list_inf_time = []
    for patch_size in patch_sizes:
        default_params_1['patch_size'] = patch_size
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=default_params_2['input_shape']),
            VisionTransformer(**default_params_1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(default_params_1['num_classes'], activation='softmax')
        ])
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
             optimizer=tfa.optimizers.AdamW(learning_rate=default_params_2['lr'], weight_decay=default_params_2['weight_decay']),
             metrics=['accuracy'])
        
        L = int((default_params_2['input_shape'][0]/default_params_1['patch_size'])**2)
        print(f"ViT with L={L} and {model.count_params()} parameters")
    
        # calculate avgerage inference time
        start_time = time.time()
        model.predict(test_sample, batch_size=1)
        stop_time1 = time.time()
        model.predict(test_sample, batch_size=1)
        stop_time2 = time.time()
        model.predict(test_sample, batch_size=1)
        stop_time3 = time.time()
        model.predict(test_sample, batch_size=1)
        stop_time4 = time.time()
        model.predict(test_sample, batch_size=1)
        stop_time5 = time.time()

        duration1 = stop_time1 - start_time
        duration2 = stop_time2 - stop_time1
        duration3 = stop_time3 - stop_time2
        duration4 = stop_time4 - stop_time3
        duration5 = stop_time5 - stop_time4
    
        inf_time = round((duration1+duration2+duration3+duration4+duration5)*200, 3)
        list_inf_time.append(inf_time)
        print(f'Inference Time: {inf_time}')
        print('*' * 79)
        
    return list_inf_time
    
    
def ViP_inf_time_check(default_params_1, default_params_2, ViP_params, patch_sizes, test_sample):
    # initialize the model 
    list_inf_time = []
    for patch_size in patch_sizes:  
        default_params_1['patch_size'] = patch_size
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=default_params_2['input_shape']),
            VisionPerformer(**{**default_params_1, **ViP_params}),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(default_params_1['num_classes'], activation='softmax')
        ])
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
             optimizer=tfa.optimizers.AdamW(learning_rate=default_params_2['lr'], weight_decay=default_params_2['weight_decay']),
             metrics=['accuracy'])
        
        L = int((default_params_2['input_shape'][0]/default_params_1['patch_size'])**2)
        print(f"ViP with L={L} and {model.count_params()} parameters")
    
        # calculate avgerage inference time
        start_time = time.time()
        model.predict(test_sample, batch_size=1)
        stop_time1 = time.time()
        model.predict(test_sample, batch_size=1)
        stop_time2 = time.time()
        model.predict(test_sample, batch_size=1)
        stop_time3 = time.time()
        model.predict(test_sample, batch_size=1)
        stop_time4 = time.time()
        model.predict(test_sample, batch_size=1)
        stop_time5 = time.time()

        duration1 = stop_time1 - start_time
        duration2 = stop_time2 - stop_time1
        duration3 = stop_time3 - stop_time2
        duration4 = stop_time4 - stop_time3
        duration5 = stop_time5 - stop_time4
    
        inf_time = round((duration1+duration2+duration3+duration4+duration5)*200, 3)
        list_inf_time.append(inf_time)
        print(f'Inference Time: {inf_time}')
        print('*' * 79)    
        
    return list_inf_time
    