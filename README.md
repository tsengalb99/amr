# amr
This is the code for adaptive resolution sequence to sequence experiments. 

'data_gen.py' is used for generating testing data.
'custom_arg.py' is the argument class that you pass in to an experiment object in 'run_exp.py'

'model.py', 'train.py', and 'datareader.py' are the model, training algorithm, and dataloader for pytorch. 

This code is threadsafe, so you can spin up multiple experiments at the same time reading from the same datafile.
