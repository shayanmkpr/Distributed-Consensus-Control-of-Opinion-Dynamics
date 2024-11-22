import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import keras
from keras import layers
from keras import regularizers


input= layers.Input(shape=(300))
#actor
action_input = layers.Dense(units=300 , activation="tanh" , kernel_regularizer=regularizers.l2(0.001))(input)
attention_layer = layers.Attention()([action_input , input])
attention_layer = layers.Concatenate()([attention_layer*action_input , action_input])
attention_layer = layers.Softmax()(attention_layer)
action_hidden = layers.LayerNormalization()(attention_layer)
action_hidden = layers.Dense(units=75 , activation="mish")(action_hidden)
action_hidden = layers.Dropout(0.1)(action_hidden)
action_hidden = layers.LayerNormalization()(action_hidden)
action_hidden = layers.Dense(units=64 , activation="mish")(action_hidden)
action_hidden = layers.Dropout(0.1)(action_hidden)
action_hidden = layers.LayerNormalization()(action_hidden)
action_hidden = layers.Dense(units=32 , activation="mish")(action_hidden)
action_hidden = layers.Dropout(0.1)(action_hidden)
action_hidden = layers.Dense(units=32 , activation="mish" )(action_hidden)
action = layers.Dense(units=3 , activation='tanh' , name = "action_output" , kernel_regularizer=regularizers.l2(0.001))(action_hidden)

#critic
critic_input = layers.Concatenate()([input , action])
critic_input = layers.Dense(64 , activation='linear' , kernel_regularizer=regularizers.l2(0.001))(critic_input)
critic_hidden = layers.LayerNormalization()(critic_input)
critic_hidden = layers.Dense(32 , activation="mish")(critic_hidden)
critic_hidden = layers.Dropout(0.1)(critic_hidden)
critic_hidden = layers.Dense(16 , activation="mish")(critic_hidden)
critic_output= layers.Dense(1 , activation="tanh", name = "critic_output" , kernel_regularizer=regularizers.l2(0.001))(critic_hidden)

actor_critic_model = keras.models.Model(inputs = input , outputs = [action , critic_output])
