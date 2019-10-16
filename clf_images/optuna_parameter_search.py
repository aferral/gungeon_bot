import traceback
import os
import optuna




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)

from optuna.trial import  Trial
import pickle
from clf_images.clasify_enemy import  train_run


optuna.logging.set_verbosity(optuna.logging.DEBUG)

def function_optimize_optuna(trial : Trial):
    train_it= 600


    stack_layers = trial.suggest_int('stack_layers', 1, 3) #2
    max_pool_layers = trial.suggest_categorical('max_pool_layers', [True, False])
    if max_pool_layers:
        max_pool_layers = stack_layers
    else:
        max_pool_layers = 0

    st_filter = int(trial.suggest_discrete_uniform('st_filter', 30, 50, 10)) #40
    inc_filter = int(trial.suggest_discrete_uniform('inc_filter', 30, 50, 10))  #50

    extra_porc = trial.suggest_int('extra_porc', 1, 4)  # 2
    input_factor = trial.suggest_categorical('input_factor', [0.5,0.25]) # 0.5


    lr = trial.suggest_categorical('learning_rate', [0.001,0.0001])
    #loss_string_options = ['cross_entropy','mse']
    #loss_string = trial.suggest_categorical('loss_string', loss_string_options)
    loss_string = 'cross_entropy'
    replace_max_pool_with_stride = trial.suggest_categorical('replace_max_pool_with_stride', [True,False])


    exp_params = {
        'tf_config' : tf.ConfigProto(allow_soft_placement = True),
        'max_pool_layers' : max_pool_layers,
        'stack_layers' : stack_layers,
        'input_factor' : input_factor,
        'extra_porc' : extra_porc,
        'lr' : lr,
        'st_filter' : st_filter,
        'inc_filter' : inc_filter,
        'loss_string' : loss_string,
        'replace_max_pool_with_stride' : replace_max_pool_with_stride
    }
    print("PARAMS : {0}".format(exp_params))

    out_dict,out_folder = train_run(train_it, save_model=True, interactive_plot=False, **exp_params)

    # save params
    metric = float(out_dict['global_F1'])

    trial.set_user_attr('out_path',out_folder)
    for k in out_dict:
        if k != 'global_F1':
            trial.set_user_attr(k,float(out_dict[k]))

    return metric




if __name__ == '__main__':

    study = optuna.create_study(study_name='clf_enemy_study', storage='sqlite:///clf_param_search.db',load_if_exists=True,direction='maximize')
    study.optimize(function_optimize_optuna, n_trials=100)


    study_data = study.trials_dataframe()
    study_data.to_csv('out_study.csv')

