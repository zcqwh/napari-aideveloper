#!/usr/bin/env python3
"""
Created on Thu Mar 24 13:13:59 2022

@author: nana
"""
import keras_metrics
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Model

#from . import keras_metrics #side package for precision, recall etc during training
global keras_metrics


class cyclicLR(tf.keras.callbacks.Callback):
    """
    Reference: https://github.com/bckenstler/CLR/blob/master/clr_callback.py
    Author: Brad Kenstler

    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = cyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = cyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super().__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


class exponentialDecay(tf.keras.callbacks.Callback):
    """
    Decrease learning rate using exponential decay
    lr = initial_lr * decay_rate ** (batch_iteration / decay_steps)
    """
    def __init__(self, initial_lr=0.01, decay_steps=100, decay_rate=0.95):
        super().__init__()

        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        self.iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_initial_lr=None, new_decay_steps=None,new_decay_rate=None):
        if new_initial_lr != None:
            self.initial_lr = new_initial_lr
        if new_decay_steps != None:
            self.decay_steps = new_decay_steps
        if new_decay_rate != None:
            self.decay_rate = new_decay_rate
        self.iterations = 0.

    def exp_decay(self):
        return self.initial_lr * self.decay_rate ** (self.iterations / self.decay_steps)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.iterations == 0:
            K.set_value(self.model.optimizer.lr, self.initial_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.exp_decay())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.exp_decay())


def get_config(cpu_nr,gpu_nr,deviceSelected,gpu_memory):
    #No GPU available, CPU selected:
    if gpu_nr==0: #and deviceSelected=="Default CPU":
        print("Adjusted options for CPU usage")
        config_gpu = tf.compat.v1.ConfigProto()
    #GPU available but user wants to use CPU
    elif gpu_nr>0 and deviceSelected=="Default CPU":
        config_gpu = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=cpu_nr,\
                inter_op_parallelism_threads=cpu_nr, allow_soft_placement=True,\
                device_count = {'CPU' : 1, 'GPU' : 0})
        print("Adjusted options for CPU usage")

    #GPU selected
    elif deviceSelected=="Single-GPU" or deviceSelected=="Multi-GPU":
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory,\
                                    allow_growth = True)
        config_gpu = tf.compat.v1.ConfigProto(allow_soft_placement=True,\
                                    gpu_options = gpu_options,\
                                    log_device_placement=False)
        if deviceSelected=="Single-GPU":
            print("Adjusted GPU options for Single-GPU usage. Set memeory fraction to "+str(gpu_memory))
        if deviceSelected=="Multi-GPU":
            print("Adjusted GPU options for Multi-GPU usage. Set memeory fraction to "+str(gpu_memory))


    return config_gpu

def model_in_out_dim(model_object,object_type):
    if object_type=="config":
    #user provided model_config:
        try: #Sequential Model
            in_dim = model_object['config'][0]['config']['batch_input_shape']
        except: #Functional Api
            in_dim = model_object['config']["layers"][0]["config"]["batch_input_shape"]
        try: #Sequential Model
            out_dim = model_object['config'][-2]['config']['units']
        except: #Functional Api
            out_dim = model_object['config']["layers"][-2]["config"]["units"]

    if object_type=="model":
        in_dim = model_object.layers[0].get_input_shape_at(0)
        out_dim = model_object.layers[-1].get_output_shape_at(0)[1]

    return in_dim,out_dim

def model_add_classes(model_keras,nr_classes):
    #Sequential or Functional API?
    model_config = model_keras.get_config()
    if "sequential" not in model_config["name"]:
        print("Loaded model is defined using functional API of Keras")
        model_api = "functional"
    if "sequential" in model_config["name"]:
        print("Loaded model is defined using the sequential API of Keras")
        model_api = "sequential"

    #get list of layer types
    layer_types_list = [l.__class__.__name__ for l in model_keras.layers]
    #get the index of the last dense layer
    ind = [name=="Dense" for name in layer_types_list]#where are dense layers
    ind = np.where(np.array(ind)==True)[0][-1]#index of the last dense layer
    last_dense_name = model_keras.layers[ind].name #name of that last dense layer
    ind = ind-1 #go even one more layer back, because we need its output

    #For functional API:
    if model_api=="functional":
        #overwrite model_keras with a new, (shortened) model
        model_keras = Model(model_keras.get_input_at(0),model_keras.layers[ind].output)
        #Add a new final dense layer
        x = Dense(nr_classes,name=last_dense_name)(model_keras.layers[-1].output)
        x = Activation('softmax',name="outputTensor")(x)
        model_keras = Model(inputs=model_keras.get_input_at(0), outputs=x)
    #For sequential API:
    elif model_api=="sequential":
        model_keras.pop()#remove final activation layer
        model_keras.pop()#remove last dense layer
        model_keras.add(Dense(nr_classes,name=last_dense_name))
        model_keras.add(Activation('softmax',name="outputTensor"))

    #Compile to reset the optimizer weights
    model_keras.compile(optimizer='adam', loss='categorical_crossentropy')

def get_metrics_tensors(metrics,nr_classes):
    """
    Function takes a list of metrics and creates fresh tensors accordingly.
    This is necessary, after re-compiling the model because the placeholder has
    to be updated
    """
    f1 = any(["auc" in str(a) for a in metrics])
    precision = any(["precision" in str(a) for a in metrics])
    recall = any(["recall" in str(a) for a in metrics])

    Metrics =  []
    if f1==True:
        for class_ in range(nr_classes):
            Metrics.append(tf.keras.metrics.AUC())
    if precision==True:
        for class_ in range(nr_classes):
            Metrics.append(tf.keras.metrics.Precision())
    if recall==True:
        for class_ in range(nr_classes):
            Metrics.append(tf.keras.metrics.Recall())
    metrics =  ['accuracy'] + Metrics
    return metrics

def model_compile(model_keras,loss_,optimizer_,learning_rate_,model_metrics,out_dim):
    print("Compiling")
    optimizer_name = optimizer_["comboBox_optimizer"].lower()
    print("optimizer_name:",optimizer_name)
    if optimizer_name=='sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=optimizer_["doubleSpinBox_lr_sgd"], momentum=optimizer_["doubleSpinBox_sgd_momentum"], nesterov=optimizer_["checkBox_sgd_nesterov"])
    elif optimizer_name=='rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=optimizer_["doubleSpinBox_lr_rmsprop"], rho=optimizer_["doubleSpinBox_rms_rho"])
    elif optimizer_name=='adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=optimizer_["doubleSpinBox_lr_adagrad"])
    elif optimizer_name=='adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=optimizer_["doubleSpinBox_lr_adadelta"], rho=optimizer_["doubleSpinBox_adadelta_rho"])
    elif optimizer_name=='adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer_["doubleSpinBox_lr_adam"], beta_1=optimizer_["doubleSpinBox_adam_beta1"], beta_2=optimizer_["doubleSpinBox_adam_beta2"], amsgrad=optimizer_["checkBox_adam_amsgrad"])
    elif optimizer_name=='adamax':
        optimizer = tf.keras.optimizers.Adamax(learning_rate=optimizer_["doubleSpinBox_lr_adamax"], beta_1=optimizer_["doubleSpinBox_adamax_beta1"], beta_2=optimizer_["doubleSpinBox_adamax_beta2"])
    elif optimizer_name=='nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=optimizer_["doubleSpinBox_lr_nadam"], beta_1=optimizer_["doubleSpinBox_nadam_beta1"], beta_2=optimizer_["doubleSpinBox_nadam_beta2"])
    else:
        print("Unknown optimizer!")
    model_keras.compile(loss=loss_,optimizer=optimizer,
                        metrics=get_metrics_tensors(model_metrics,out_dim))
    print("Compiling DONE")

    return 1

def model_get_trainable_list(model_keras):
    params = [model_keras.layers[i].count_params() for i in range(len(model_keras.layers))]
    #Only count layers that have parameters
    l_para_not_zero = np.where(np.array(params)>0)[0] #indices of layers that have parameters
    #Original trainable states of layers with parameters
    trainable_list = [model_keras.layers[i].trainable for i in l_para_not_zero]
    layer_names_list = [model_keras.layers[i].__class__.__name__  for i in l_para_not_zero]
    return trainable_list,layer_names_list

def get_dropout(model_keras):
    #Funktion takes a keras model and gives a list of the dropout values.
    l_names = [model_keras.layers[i].__class__.__name__ for i in range(len(model_keras.layers))]
    ind = ["Dropout" in x for x in l_names]
    ind = np.where(np.array(ind)==True)[0] #at which indices are dropout layers?
    do_list = [model_keras.layers[i].rate for i in ind]
    return do_list

def get_optimizer_name(model_keras):
    optimizer = model_keras.optimizer.get_config()["name"]
    # optimizer = str(model_keras.optimizer)
    # optimizer = optimizer.split("<keras.optimizer_v2.")[1].split(" object at")[0]
    # optimizer = optimizer.split(".")[1]
    if optimizer in ['SGD','RMSprop','Adagrad',"Adamax",'Adadelta','Adam','Nadam']:
        return optimizer
    else:
        return 0

def get_optimizer_settings():
    """
    Default optimizer settings (taken from https://keras.io/api/optimizers)
    """
    d = {}
    d["comboBox_optimizer"] = "Adam"

    d["doubleSpinBox_lr_sgd"] = 0.01
    d["doubleSpinBox_sgd_momentum"] = 0.0
    d["checkBox_sgd_nesterov"] = False

    d["doubleSpinBox_lr_rmsprop"] = 0.001
    d["doubleSpinBox_rms_rho"] = 0.9

    d["doubleSpinBox_lr_adam"] = 0.001
    d["doubleSpinBox_adam_beta1"] = 0.9
    d["doubleSpinBox_adam_beta2"] = 0.999
    d["checkBox_adam_amsgrad"] = False

    d["doubleSpinBox_lr_nadam"] = 0.002
    d["doubleSpinBox_nadam_beta1"] = 0.9
    d["doubleSpinBox_nadam_beta2"] = 0.999

    d["doubleSpinBox_lr_adadelta"] = 1.0
    d["doubleSpinBox_adadelta_rho"] = 0.95

    d["doubleSpinBox_lr_adagrad"] = 0.01

    d["doubleSpinBox_lr_adamax"] = 0.002
    d["doubleSpinBox_adamax_beta1"] = 0.9
    d["doubleSpinBox_adamax_beta2"] = 0.999

    return d

def get_cyclStepSize(SelectedFiles,step_size,batch_size):
    ind = [selectedfile["TrainOrValid"] == "Train" for selectedfile in SelectedFiles]
    ind = np.where(np.array(ind)==True)[0]
    SelectedFiles_train = np.array(SelectedFiles)[ind]
    SelectedFiles_train = list(SelectedFiles_train)
    nr_events_train_total = np.sum([int(selectedfile["nr_events_epoch"]) for selectedfile in SelectedFiles_train])

    cycLrStepSize = step_size*int(np.round(nr_events_train_total / batch_size))#number of steps in one epoch
    return cycLrStepSize

def model_change_trainability(model_keras,trainable_new,model_metrics,out_dim,loss_,optimizer_,learning_rate_):
    print("Line 83")
    #Function takes a keras model and list of trainable states.
    #The last n layers, which have parameters (not activation etc.) are set to 'not trainable'
    params = [model_keras.layers[i].count_params() for i in range(len(model_keras.layers))]
    #Only count layers that have parameters
    l_para_not_zero = np.where(np.array(params)>0)[0] #indices of layers that have parameters
    nr_layers = len(l_para_not_zero) #total nr. of layers of model with parameters
    #check that the number of layers in model is equal to the len(trainable_new)
    assert nr_layers==len(trainable_new)
    #Original trainable states of layers with parameters
    trainable_now = [model_keras.layers[i].trainable for i in l_para_not_zero]
    if trainable_now == trainable_new:#they are already equal
        text = "No need to change trainable states of model."
    else:
        print("Change trainable states of model.")
        #Now change the trainability of each layer with parameters to the state given in trainable_states
        for i in range(nr_layers):
            layer_index = l_para_not_zero[i]
            model_keras.layers[layer_index].trainable = trainable_new[i]
        #Model has to be recompiled in order to see any any effect of the change
        model_compile(model_keras,loss_,optimizer_,learning_rate_,model_metrics,out_dim)

        text = []
        model_keras.summary(print_fn=text.append)
        text = "\n".join(text)
    return text

#Define some custom metrics which will allow to use precision, recall, etc during training
def get_custom_metrics():
    custom_metrics = []
    #Keep keras_metrics to maintain compatibiliy with AID_0.2.x versions
    custom_metrics.append(keras_metrics.categorical_f1_score())
    custom_metrics.append(keras_metrics.categorical_precision())
    custom_metrics.append(keras_metrics.categorical_recall())
    custom_metrics = {m.__name__: m for m in custom_metrics}
    custom_metrics["sin"] = K.sin
    custom_metrics["abs"] = K.abs
    return custom_metrics

def get_lr_dict(learning_rate_const_on,learning_rate_const,
                    learning_rate_cycLR_on,cycLrMin,cycLrMax,
                    cycLrMethod,cycLrStepSize,
                    learning_rate_expo_on,
                    expDecInitLr,expDecSteps,expDecRate,cycLrGamma):
    lr_dict = pd.DataFrame()
    lr_dict["learning_rate_const_on"] = learning_rate_const_on,
    lr_dict["learning_rate_const"] = learning_rate_const,
    lr_dict["learning_rate_cycLR_on"] = learning_rate_cycLR_on,
    lr_dict["cycLrMin"] = cycLrMin,
    lr_dict["cycLrMax"] = cycLrMax,
    lr_dict["cycLrMethod"] = cycLrMethod,
    lr_dict["cycLrStepSize"] = cycLrStepSize,
    lr_dict["expDecInitLr"] = expDecInitLr,
    lr_dict["expDecInitLr"] = expDecInitLr,
    lr_dict["expDecSteps"] = expDecSteps,
    lr_dict["expDecRate"] = expDecRate,
    lr_dict["cycLrGamma"] = cycLrGamma,
    return lr_dict

def get_lr_callback(learning_rate_const_on,learning_rate_const,
                    learning_rate_cycLR_on,cycLrMin,cycLrMax,
                    cycLrMethod,cycLrStepSize,
                    learning_rate_expo_on,
                    expDecInitLr,expDecSteps,expDecRate,cycLrGamma):

    if learning_rate_const_on==True:
        return None
    elif learning_rate_cycLR_on==True:
        return cyclicLR(base_lr=cycLrMin,max_lr=cycLrMax,step_size=cycLrStepSize,mode=cycLrMethod,gamma=cycLrGamma)
    elif learning_rate_expo_on==True:
        return exponentialDecay(initial_lr=expDecInitLr, decay_steps=expDecSteps, decay_rate=expDecRate)

def change_dropout(model_keras,do,model_metrics,out_dim,loss_,optimizer_,learning_rate_):
    #Funktion takes a keras model and changes the dropout.
    #do = float (0 to 1) or list (with values from 0 to 1)
    #if do is a list: len(do) has to be equal to the nr. of dropout layers in the model
    #if do is a float: this value is then used for all droput layers
    model_keras._make_train_function()
    model_weights = model_keras.get_weights() # Store weights
    optimizer_weights = model_keras.optimizer.get_weights() # Store optimizer values

    l_names = [model_keras.layers[i].__class__.__name__ for i in range(len(model_keras.layers))]
    ind = ["Dropout" in x for x in l_names]
    ind = np.where(np.array(ind)==True)[0] #at which indices are dropout layers?
    if len(ind)==0:#if there are no dropout layers, return
        return 0
    if type(do)==float:
        #set all dropout layers to do
        for i in range(len(ind)):
            model_keras.layers[ind[i]].rate = do
    elif type(do)==list:
        #check that the length of the list equals the nr. of dropout layers
        if len(do)!=len(ind):
            return 0
        else:
            for i in range(len(ind)):
                model_keras.layers[ind[i]].rate = do[i]

    #Only way that it actually has an effect is to clone the model, compile and reload weights
    model_keras = tf.keras.models.clone_model(model_keras) # If I do not clone, the new rate is never used. Weights are re-init now.
    model_compile(model_keras,loss_,optimizer_,learning_rate_,model_metrics,out_dim)

    model_keras._make_train_function()
    model_keras.set_weights(model_weights) # Set weights
    model_keras.optimizer.set_weights(optimizer_weights)#Set optimizer values
    return 1
