Problem Statement:

Using PyTorch, implement a version of the CNN network architecture to train and test a handwritten digit recognition
system. You can read the paper ”Backpropagation Applied to Handwritten Zip Code
Recognition” by LeCun et al. 1989 for more details, but your architecture will not follow exactly
what was mentioned in the paper. Use the MNIST dataset to train and test the system. Be sure
to divide the data into a single training, validation and testing set. Note that you do not need to
resize the inputs to size 16 × 16. The baseline system should use the following: Glorot
initialization, ReLU activations, mini-batch stochastic gradient descent with momentum (β = 0.9),
model selection and a cross-entropy loss function. Use a learning rate scheduler to adjust the
learning rate by 10% every 10 epochs, starting with a learning rate of 0.05. You are encouraged
to use Carbonate Deep Learning cluster to develop your system, since they have GPUs which
will improve the training process. Generate learning curves for the validation and training set.
Discuss whether this baseline system overfits, underfits or reasonably fits the validation data.
Test this baseline system with the testing data and report the accuracy and show a confusion
matrix. Submit your solution to this part of the problem in a Jupyter notebook document named:
baseline.ipynb. Separately incorporate the following changes to the baseline model (e.g. do not
do dropout and RMSprop at the same time), and generate learning curves along with a
confusion matrix for the testing set. Create and submit 3 separate Jupyter notebook files, one
for each update to test (e.g., baseline-dropout.ipynb, baseline-batch.ipynb,
baseline-optim.ipynb).

1. Add dropout using drop rates of 0.25, 0.5 and 0.75, respectively
2. Incorporate batch normalization before each convolutional hidden layer.
3. Separately train using RMSProp, ADAM, and Nesterov optimizers