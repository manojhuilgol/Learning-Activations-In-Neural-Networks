# Learning_Activations_in_Neural_Networks

## Data:

Three datasets are used for running the experiments:
- MNIST dataset 
- IRIS
- Bank Note

All the datasets are downloaded from the URL link available online automatically when running the main() method.

## Run:
The main() method needs to be run in order to execute the experiments.
The hyperparameters can be changed from inside the main() method:
```
hyperparams: input_dims, hidden_dims, output_dims, lr_rate, epochs.
```

The two cases as mentioned in the report can be used with the below specifications to replicate the results as in the report:
```
case_1: {"mean:1", "std_dev":0.5}
case_2: {"mean:-2", "std_dev":1}
```

## file descriptions:

- The [main.py](https://github.com/manojhuilgol/Learning-Activations-In-Neural-Networks/blob/main/main.py) provides needs to be run which contains the dataset preprocessing and experiment specifications.

- The [neuralnetwork.py](https://github.com/manojhuilgol/Learning-Activations-In-Neural-Networks/blob/main/neuralnetwork.py) contains the implementation of Multilayer Perceptron.

- The [utils.py](https://github.com/manojhuilgol/Learning-Activations-In-Neural-Networks/blob/main/utils.py) is provided with common methods/functions used during the training process.
 
- The [utils.py](https://github.com/manojhuilgol/Learning-Activations-In-Neural-Networks/blob/main/visualize_results.ipynb) contains the Jupyter Notebook with the plots of the results and visualization for all the experiments.


## Results:
The results with the trained metrics are saved in the [results](https://github.com/manojhuilgol/Learning-Activations-In-Neural-Networks/blob/main/results) folder.


## Contact:
**Manoj Huilgol** (manojhuilgol1994@gmail.com)
