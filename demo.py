from model import *
from rbf import *
from MDE_LBFGS import *
import pandas as pd


# Goal: Classify rocks or iron things (danger) given a 60 dimensional vector from the sensors of a sonar
# The dataset is very poor (only 200 example ).
# We train on 80% and validate on the 20%. Given the scarcity of data, overfitting is very hard to avoid


# Read Dataset
data = pd.read_csv("sonar.csv")
print ("Data shape: "+str(data.shape))


# ---DATA PREPROCESSING---

# Remove missing values and shuffle
data = data.dropna()
data = data.sample(frac=1.0)

# Get label column and remove from train set
data_label = data['R']
data = data.drop(columns=['R'])

# Standardize the train set
data = (data - data.mean())/data.std()


# Build data for training: Make train and validation set

N = len(data.values)

# Spilt data 20% validation - 80% train
n = int(N/5)
x_val = data.values[0:n]
y_val = data_label.values[0:n]
x_train = data.values[n:]
y_train = data_label.values[n:]

# Get the label in 0/1 format
y_train = np.array([0 if y_train[i] == 'R' else 1 for i in range(len(y_train))])
y_val = np.array([0 if y_val[i] == 'R' else 1 for i in range(len(y_val))])


x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_val = x_val.astype('float32')
y_val = y_val.astype('float32')
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)


def rescale_hyperparameters(params):

    x = np.array(params)

    # l1 regularizer
    x[0] = 10**x[0]

    # learning rate
    x[1] = 10**x[1]

    # momentum
    x[2] = x[2]/10

    # decay
    x[3] = 10**x[3]

    #dropout
    x[4] = x[4]/10

    return x


# Define the model and the data
f = Model(x_train, y_train, x_val, y_val, batch_size=32, epochs=100)

# The base knowledge of hyper-parameters.
# Scaling all of them can help to prevent bad conditioning in the surrogate matrix
x = np.array([[-3, -4, 8.8, -2, 5], [-2, -3, 9.8, -2, 5], [-2, -4, 9, -1.5, 7]])
y = np.array([0.7537, 0.5596, 0.7887])

print("Best already known : "+str(min(y)))

# Set up the bounds for reasonably good choices of hyper parameters
bounds = ((-3, -1), (-3, -1), (8, 10), (-6, -1), (1, 10))

best_y = y[0]
best_x = x[0]

# Define the parameter of RBF fit
gamma = 1

# Define the optimizer for the bumpiness problem
alg = MDE_LBFGS(x.shape[1], bounds, pop_size=10, gen=20)


for i in range(20):

    # Set the aspiration level. With p = 25% we make a global exploration
    aspiration_level = 0
    if np.random.rand() < 0.25:
        aspiration_level = -np.inf

    # Define the surrogate model for the iteration
    surrogate = RBF(x, y, gamma)
    bumpiness_problem = BumpinessProblem(x.shape[1], bounds, aspiration_level, surrogate)

    print ("\n\n\nIteration: " + str(i) + " \nCondition number of the surrogate matrix: " + str(np.linalg.cond(surrogate.phi)))

    # Get the minimizer of the bumpiness problem
    x_new = alg.solve(bumpiness_problem)
    print ("Bumpiness min: " + str(bumpiness_problem(x_new)) + " at x= " + str(rescale_hyperparameters(x_new)))

    y_new = f.evaluate_model(rescale_hyperparameters(x_new))[1]

    # So we add the new value
    y = np.append(y, y_new)
    x = np.append(x, [x_new], axis=0)
    if y_new < best_y:
        best_y = y_new
        best_x = x_new

print ("\n\n Best = "+str(best_y)+" with hyper-parameters "+str(rescale_hyperparameters(best_x)))



