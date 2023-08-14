# Simple introduction for model construction https://www.pythonf.cn/read/72352

import time
start_time = time.time()
# print('start_time: ', start_time)

# import PySimpleGUI as sg

import matplotlib
matplotlib.use('TkAgg')

# matplotlib.use('Agg')  # only save figs instead of showing
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasAgg
# import matplotlib.backends.tkagg as tkagg
import tkinter as Tk


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import math
import pandas as pd
from math import sqrt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sys
import pickle

import torch
from torch import nn, optim

from torchvision import transforms
from collections import OrderedDict


# Use GPU if it's available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## Transforms features by scaling each feature to a given range.
## This estimator scales and translates each feature individually such that it is in the given range on the training set, i.e. between zero and one.
## This transformation is often used as an alternative to zero mean, unit variance scaling.
## fit(X[, y])	Compute the minimum and maximum to be used for later scaling.
## transform(X)	Scaling features of X according to feature_range.
## fit_transform(X[, y])	Fit to data, then transform it.
## inverse_transform(X)	Undo the scaling of X according to feature_range.
scaler1 = MinMaxScaler()  
scaler2 = MinMaxScaler()  
scaler3 = MinMaxScaler()

no_of_input_nodes = 2
no_of_output_nodes = 5
# porosity, flow resistance, tortuosity, characteristic viscous length, characteristic thermal length

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100


df_0 = pd.read_excel('JCA_data_array_ForTrainingV3.xlsx', sheet_name='sheet')
datafile_0 = df_0.values                  ## stored data from xlsx file
out_var_datafile_0 = datafile_0[:,range(2,7)]      ## range(2,7) 2 3 4 5 6           stored output_variable (4th column) from xlsx file
out_var_datafile_0 = out_var_datafile_0.reshape((-1,no_of_output_nodes))    ## one column with unknown no. of row
in_var_datafile_0 = datafile_0[:,range(0,2)]      ## range(2,7) 2 3 4 5 6           stored output_variable (4th column) from xlsx file
in_var_datafile_0 = in_var_datafile_0.reshape((-1,no_of_input_nodes))    ## one column with unknown no. of row
scaler1.fit(datafile_0)
scaler2.fit(out_var_datafile_0)
scaler3.fit(in_var_datafile_0)

df_1 = pd.read_excel('training_dataset.xlsx', sheet_name='Sheet1')
datafile_1 = df_1.values
print(datafile_1.shape)
scaler_datafile_1 = scaler1.transform(datafile_1)
X_train = scaler_datafile_1[:, range(0, 2)]
y_train = scaler_datafile_1[:, range(2, 7)]

df_2 = pd.read_excel('validation_dataset.xlsx', sheet_name='Sheet1')
datafile_2 = df_2.values
print(datafile_2.shape)
scaler_datafile_2 = scaler1.transform(datafile_2)
X_validation = scaler_datafile_2[:, range(0, 2)]
y_validation = scaler_datafile_2[:, range(2, 7)]

df_3 = pd.read_excel('test_dataset.xlsx', sheet_name='Sheet1')
datafile_3 = df_3.values
print(datafile_3.shape)
scaler_datafile_3 = scaler1.transform(datafile_3)
X_test = scaler_datafile_3[:, range(0, 2)]
y_test = scaler_datafile_3[:, range(2, 7)]

plt.figure()
plt.scatter(scaler3.inverse_transform(X_validation)[:, 0], scaler3.inverse_transform(X_validation)[:, 1], s=20, marker='o', c="blue")
plt.scatter(scaler3.inverse_transform(X_train)[:, 0], scaler3.inverse_transform(X_train)[:, 1], s=20, marker='^', c="red")
plt.scatter(scaler3.inverse_transform(X_test)[:, 0], scaler3.inverse_transform(X_test)[:, 1], s=20, marker='o', c="green")
# plt.plot(scaler2.inverse_transform(predicted_on_X_test)[:, 0], 'b*', markersize=12, label='Predictions')
plt.savefig('Dataset_distribution_figure.png', bbox_inches='tight')

input_dim = 2  ## = no. of input variables columns
output_dim = no_of_output_nodes  ## = no. of output variables columns
from collections import OrderedDict

# ############     model without dropout     #####################
# nodes_hidden_1 = 20
# nodes_hidden_2 = 20
# ## nn.Linear() is fully connected layer
# model = nn.Sequential(OrderedDict([
#                         ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
#                         ('relu', nn.ReLU()),
#                         ('fc2', nn.Linear(nodes_hidden_1, nodes_hidden_2)),
#                         ('relu', nn.ReLU()),
#                         ('fc3', nn.Linear(nodes_hidden_2, output_dim)),
#                         ]))


############     model with dropout - 3 layers    #####################
####             dropout_prob leads to variations in mse curve      #########
# construct training model by using pytorch
dropout_prob = 0.0
nodes_number= 200
nodes_hidden_1 = nodes_number  # 150
nodes_hidden_2 = nodes_number
nodes_hidden_3 = nodes_number
nodes_hidden_4 = nodes_number
nodes_hidden_5 = nodes_number
## nn.Linear() is fully connected layer

# 3layer
model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(dropout_prob)),
    ('fc2', nn.Linear(nodes_hidden_1, nodes_hidden_2)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(dropout_prob)),
    ('fc3', nn.Linear(nodes_hidden_2, nodes_hidden_3)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(dropout_prob)),
    ('fc4', nn.Linear(nodes_hidden_3, output_dim)),
]))

# 5layer

# model = nn.Sequential(OrderedDict([
#     ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc2', nn.Linear(nodes_hidden_1, nodes_hidden_2)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc3', nn.Linear(nodes_hidden_2, nodes_hidden_3)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc4', nn.Linear(nodes_hidden_3, nodes_hidden_4)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc4', nn.Linear(nodes_hidden_4, nodes_hidden_5)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc5', nn.Linear(nodes_hidden_5, output_dim)),
# ]))

# 4 layer
# model = nn.Sequential(OrderedDict([
#     ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc2', nn.Linear(nodes_hidden_1, nodes_hidden_2)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc3', nn.Linear(nodes_hidden_2, nodes_hidden_3)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc4', nn.Linear(nodes_hidden_3, nodes_hidden_4)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc4', nn.Linear(nodes_hidden_4, output_dim)),
# ]))

# 2layer
# model = nn.Sequential(OrderedDict([
#     ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc2', nn.Linear(nodes_hidden_1, nodes_hidden_2)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc3', nn.Linear(nodes_hidden_2, output_dim)),
# ]))

# 1layer
# model = nn.Sequential(OrderedDict([
#     ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc2', nn.Linear(nodes_hidden_1, output_dim)),
# ]))

# ############     model with dropout - 2 layers     #####################
# ####             dropout_prob leads to variations in mse curve    ###########
# dropout_prob = 0.1           # 0.5 - used in nvidia model-behavioural cloning
# nodes_hidden_1 = 50
# nodes_hidden_2 = 50
# ## nn.Linear() is fully connected layer
# model = nn.Sequential(OrderedDict([
#                         ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
#                         ('relu', nn.ReLU()),
#                         ('dropout', nn.Dropout(dropout_prob)),
#                         ('fc2', nn.Linear(nodes_hidden_1, nodes_hidden_2)),
#                         ('relu', nn.ReLU()),
#                         ('dropout', nn.Dropout(dropout_prob)),
#                         ('fc3', nn.Linear(nodes_hidden_2, output_dim)),
#                         ]))


print(model)
# model.double()
# print(X_train)
print(X_train.shape, y_train.shape)

f = open("Performance_ANNmodel_Output"+".dat", 'w')
f.write("X_training_dataset_shape")
f.write('\n')
f.write(str(X_train.shape))
f.write('\n')
f.write("X_validation_dataset_shape")
f.write('\n')
f.write(str(X_validation.shape))
f.write('\n')
f.write("X_test_dataset_shape")
f.write('\n')
f.write(str(X_test.shape))
f.write('\n')

criterion = nn.MSELoss()
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
print(device)
## move model to gpu if available, else cpu
# model.to(device)


epochs = 5000  # 5000
# Convert numpy array to torch Variable
# inputs = torch.from_numpy(X_train).requires_grad_()
# labels = torch.from_numpy(y_train)
inputs = torch.Tensor((X_train))
labels = torch.Tensor((y_train))
inputs_validation = torch.Tensor((X_validation))
labels_validation = torch.Tensor((y_validation))
running_loss = []
running_loss_validation = []
for epoch in range(epochs):
    epoch += 1

    #################   train the model   ######################
    model.train()  # prep model for training
    # Clear gradients w.r.t. parameters, else gradients will be added up with every previous pass
    optimizer.zero_grad()
    # Forward to get output
    outputs = model(inputs)
    # Calculate Loss
    loss = criterion(outputs, labels)  ## mean squared error
    # Getting gradients w.r.t. parameters
    loss.backward()
    # Updating parameters
    optimizer.step()  ## take a step with optimizer to update the weights
    running_loss.append(loss.item())

    # ###############    validate the model (not showing fluctuations)      ###################
    # # Turn off gradients for validation, saves memory and computations
    # with torch.no_grad():
    #     ## this turns off dropout for evaluation mode of model
    #     model.eval()      # prep model for evaluation
    #     outputs_validation = model(inputs_validation)
    #     loss_validation = criterion(outputs_validation, labels_validation)
    #     running_loss_validation.append(loss_validation.item())

    # ###############    validate the model (showing fluctuations)      ###################
    outputs_validation = model(inputs_validation)
    loss_validation = criterion(outputs_validation, labels_validation)
    running_loss_validation.append(loss_validation.item())

    print('epoch: {}, mse_loss: {:.6f}, mse_loss_validation: {:.6f}'.format(epoch, loss.item(), loss_validation.item()))
    # print(mean_squared_error(outputs_validation,labels_validation))


# save the model, as weights & parameters are stored in model.state_dict()
# print(model.state_dict().keys())
# print(model.state_dict())
#### torch.save(model.state_dict(), 'checkpoint-epochs-{}.pth'.format(epochs))
torch.save(model.state_dict(), 'model.pkl')
# # load the saved model at particular epochs to compare
state_dict = torch.load('model.pkl')
# load the saved model
#### state_dict = torch.load('checkpoint-epochs-{}.pth'.format(epochs))
# state_dict = torch.load('checkpoint.pth')
# state_dict = torch.load('checkpoint-simple_waveguide_neff_pytorch_1_epochs-5000.pth')
model.load_state_dict(state_dict)

# Purely inference
# predicted_on_X_train = model(torch.Tensor(X_train).requires_grad_()).data.numpy()
# predicted_on_X_validation = model(torch.Tensor(X_validation).requires_grad_()).data.numpy()
# predicted_on_X_test = model(torch.Tensor(X_test).requires_grad_()).data.numpy()
with torch.no_grad():
    ## this turns off dropout for evaluation mode of model
    model.eval()
    predicted_on_X_train = model(torch.Tensor(X_train)).data.numpy()
    predicted_on_X_validation = model(torch.Tensor(X_validation)).data.numpy()
    predicted_on_X_test = model(torch.Tensor(X_test)).data.numpy()
    # print(predicted)

end_time = time.time()
print('end_time: ', end_time)
print('time taken to train in sec: ', (end_time - start_time))

## make axis bold
plt.rcParams.update({'font.size': 26})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.figsize"] = (8.5,8)
plt.rcParams["font.family"] = "Times New Roman"

mse_training_interval = 10
mse_validation_interval = 10
running_loss = running_loss[::mse_training_interval]
running_loss_index = [i for i in range(1, epochs, mse_training_interval)]
running_loss_validation = running_loss_validation[::mse_validation_interval]
running_loss_validation_index = [i for i in range(1, epochs, mse_validation_interval)]
print('mse lengths: ', len(running_loss), len(running_loss_validation))
# print('running_loss_index: ', running_loss_index)
# print('running_loss_validation_index: ', running_loss_validation_index)


###############################################################
#################   plotting graphs together - porosity  ################
###############################################################

# plt.suptitle('JCA - porosity - (epochs-{}) - pyTorch'.format(epochs), fontsize=25,
#              color='r', fontweight='bold')  ## giving title on top of all subplots

plt.figure()
plt.plot(running_loss_index, running_loss, 'r-', linewidth=3, label='MSE_loss_train')
plt.plot(running_loss_validation_index, running_loss_validation, 'b-', linewidth=3, label='MSE_loss_validation')
# plt.legend(loc='best', fontsize=10)
# plt.xlabel('epochs#', fontsize=15)
plt.legend()
plt.ylabel('MSE loss')
plt.xlabel('Epochs')
plt.savefig('Porosity_MES_Epochs.png', bbox_inches='tight')


plt.figure()
# Plot true data
plt.plot(scaler2.inverse_transform(y_test)[:, 0], 'ro', markersize=12, label='Targets')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_test)[:, 0], 'b*', markersize=12, label='Predictions')
# Legend and plot
# plt.legend(loc='best', fontsize=10)
plt.xlabel('Samples')
plt.ylabel('Porosity')
# plt.legend(loc='best', fontsize=10)
plt.legend(loc='best', fontsize=24)
plt.savefig('Porosity_TestData.png', bbox_inches='tight')

plt.figure()
x_major_locator=MultipleLocator(0.1)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.1)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为1的倍数
xx = scaler2.inverse_transform(y_train)[:, 0]
yy = scaler2.inverse_transform(predicted_on_X_train)[:, 0]
print('JCA - porosity - r2_score')
print(r2_score(xx, yy))  #r2_score(y_true, y_pred)
print('JCA - porosity - mean_squared_error')
print(mean_squared_error(xx, yy))  #mean_squared_error(y_true, y_pred)
print('JCA - porosity - mean_absolute_error')
print(mean_absolute_error(xx, yy))  #mean_absolute_error(y_true, y_pred)
print('JCA - porosity - mean_absolute_percentage_error')
print(mean_absolute_percentage_error(xx, yy))  #mean_absolute_percentage_error(y_true, y_pred)

xx_validation = scaler2.inverse_transform(y_validation)[:, 0]
yy_validation = scaler2.inverse_transform(predicted_on_X_validation)[:, 0]
xx_test = scaler2.inverse_transform(y_test)[:, 0]
yy_test = scaler2.inverse_transform(predicted_on_X_test)[:, 0]

print('JCA - porosity_validation - mean_absolute_percentage_error')
print(mean_absolute_percentage_error(xx_validation, yy_validation))  #mean_absolute_percentage_error(y_true, y_pred)
print('JCA - porosity_test - mean_absolute_percentage_error')
print(mean_absolute_percentage_error(xx_test, yy_test))  #mean_absolute_percentage_error(y_true, y_pred)

bubble_plot_line_x1y1 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
bubble_plot_line_x2y2 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx, yy, label=r'Porosity', marker='o', facecolors='none', edgecolors='red', s=140)  # r'$\mu = 0, \sigma^2 = 1$'
# plt.legend(loc='best')
plt.title('Porosity')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Porosity_Training.png', bbox_inches='tight')

plt.figure()
x_major_locator=MultipleLocator(0.1)
y_major_locator=MultipleLocator(0.1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
bubble_plot_line_x1y1 = [min(np.minimum(xx_validation, yy_validation)), max(np.maximum(xx_validation, yy_validation))]
bubble_plot_line_x2y2 = [min(np.minimum(xx_validation, yy_validation)), max(np.maximum(xx_validation, yy_validation))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx_validation, yy_validation, label='Porosity', marker='o', facecolors='none', edgecolors='blue', s=140)
# plt.legend(loc='best')
plt.title('Porosity')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Porosity_Validation.png', bbox_inches='tight')

plt.figure()
x_major_locator=MultipleLocator(0.1)
y_major_locator=MultipleLocator(0.1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
bubble_plot_line_x1y1 = [min(np.minimum(xx_test, yy_test)), max(np.maximum(xx_test, yy_test))]
bubble_plot_line_x2y2 = [min(np.minimum(xx_test, yy_test)), max(np.maximum(xx_test, yy_test))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx_test, yy_test, label='Porosity', marker='o', facecolors='none', edgecolors='blue', s=140)
# plt.legend(loc='best')
plt.title('Porosity')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Porosity_Test.png', bbox_inches='tight')

# print()
# print("o/p of test set:           \n", (scaler2.inverse_transform(y_test)[:, 0]))
# print("predicted o/p of test set: \n", (scaler2.inverse_transform(predicted_on_X_test)[:, 0]))
# print("mse_test_set: ", mean_squared_error(y_test, predicted_on_X_test))

###############################################################
#################   plotting graphs together - sigma  ################
###############################################################

plt.figure()
# plt.suptitle('JCA - flow resistance - (epochs-{}) - pyTorch'.format(epochs), fontsize=25,
#              color='r', fontweight='bold')  ## giving title on top of all subplots

# plt.subplot(231)
plt.plot(running_loss_index, running_loss, 'r-', linewidth=3, label='MSE_loss_train')
plt.plot(running_loss_validation_index, running_loss_validation, 'b-', linewidth=3, label='MSE_loss_validation')
# plt.legend(loc='best', fontsize=10)
# plt.xlabel('epochs#', fontsize=15)
plt.legend()
plt.ylabel('MSE loss')
plt.xlabel('Epochs')
plt.savefig('Flow resistance_MES_Epochs.png', bbox_inches='tight')


plt.figure()
# plt.subplot(234)
# Plot true data
plt.plot(scaler2.inverse_transform(y_test)[:, 1], 'ro', markersize=12, label='Targets')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_test)[:, 1], 'b*', markersize=12, label='Predictions')
# Legend and plot
# plt.legend(loc='best', fontsize=10)
plt.xlabel('Samples')
plt.ylabel('Flow resistance')
# plt.legend(loc='best', fontsize=10)
# plt.legend(loc='best', fontsize=15)
ax=plt.gca()
plt.legend(loc='best', fontsize=24)
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='y')
plt.savefig('Flow resistance_TestData.png', bbox_inches='tight')


plt.figure()
# plt.subplot(235)
xx = scaler2.inverse_transform(y_train)[:, 1]
yy = scaler2.inverse_transform(predicted_on_X_train)[:, 1]
print('JCA - flow resistance  - r2_score')
print(r2_score(xx, yy))  #r2_score(y_true, y_pred)
print('JCA - flow resistance  - mean_squared_error')
print(mean_squared_error(xx, yy))  #mean_squared_error(y_true, y_pred)
print('JCA - flow resistance  - mean_absolute_error')
print(mean_absolute_error(xx, yy))  #mean_absolute_error(y_true, y_pred)
print('JCA - flow resistance  - mean_absolute_percentage_error')
print(mean_absolute_percentage_error(xx, yy))  #mean_absolute_percentage_error(y_true, y_pred)
plt.figure()
x_major_locator=MultipleLocator(30000)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(30000)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
# #ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='both')
xx_validation = scaler2.inverse_transform(y_validation)[:, 1]
yy_validation = scaler2.inverse_transform(predicted_on_X_validation)[:, 1]
xx_test = scaler2.inverse_transform(y_test)[:, 1]
yy_test = scaler2.inverse_transform(predicted_on_X_test)[:, 1]

print('JCA - flow resistance_validation - mean_absolute_percentage_error')
print(mean_absolute_percentage_error(xx_validation, yy_validation))  #mean_absolute_percentage_error(y_true, y_pred)
print('JCA - flow resistance_test - mean_absolute_percentage_error')
print(mean_absolute_percentage_error(xx_test, yy_test))  #mean_absolute_percentage_error(y_true, y_pred)

bubble_plot_line_x1y1 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
bubble_plot_line_x2y2 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx, yy, label='Flow resistance', marker='o', facecolors='none', edgecolors='red', s=140)
# plt.legend(loc='best')
plt.title('Flow resistance')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Flow resistance_Training.png', bbox_inches='tight')

plt.figure()
x_major_locator=MultipleLocator(30000)
y_major_locator=MultipleLocator(30000)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='both')
bubble_plot_line_x1y1 = [min(np.minimum(xx_validation, yy_validation)), max(np.maximum(xx_validation, yy_validation))]
bubble_plot_line_x2y2 = [min(np.minimum(xx_validation, yy_validation)), max(np.maximum(xx_validation, yy_validation))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx_validation, yy_validation, label='Flow resistance', marker='o', facecolors='none', edgecolors='blue', s=140)
# plt.legend(loc='best')
plt.title('Flow resistance')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Flow resistance_Validation.png', bbox_inches='tight')

plt.figure()
x_major_locator=MultipleLocator(50000)
y_major_locator=MultipleLocator(50000)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='both')
bubble_plot_line_x1y1 = [min(np.minimum(xx_test, yy_test)), max(np.maximum(xx_test, yy_test))]
bubble_plot_line_x2y2 = [min(np.minimum(xx_test, yy_test)), max(np.maximum(xx_test, yy_test))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx_test, yy_test, label='Flow resistance', marker='o', facecolors='none', edgecolors='blue', s=140)
# plt.legend(loc='best')
plt.title('Flow resistance')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Flow resistance_Test.png', bbox_inches='tight')

# print()
# print("o/p of test set:           \n", (scaler2.inverse_transform(y_test)[:, 1]))
# print("predicted o/p of test set: \n", (scaler2.inverse_transform(predicted_on_X_test)[:, 1]))
# print("mse_test_set: ", mean_squared_error(y_test, predicted_on_X_test))


###############################################################
#################   plotting graphs together -  tortuosity  \alpha_{\infty}  ################
###############################################################

plt.figure()
# plt.suptitle('JCA - tortuosity - (epochs-{}) - pyTorch'.format(epochs), fontsize=25,
#              color='r', fontweight='bold')  ## giving title on top of all subplots

# plt.subplot(231)
plt.plot(running_loss_index, running_loss, 'r-', linewidth=3, label='MSE_loss_train')
plt.plot(running_loss_validation_index, running_loss_validation, 'b-', linewidth=3, label='MSE_loss_validation')
# plt.legend(loc='best', fontsize=10)
plt.legend()
plt.ylabel('MSE loss')
plt.xlabel('Epochs')
plt.savefig('Tortuosity_MES_Epochs.png', bbox_inches='tight')

plt.figure()
# plt.subplot(234)
# Plot true data
plt.plot(scaler2.inverse_transform(y_test)[:, 2], 'ro', markersize=12, label='Targets')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_test)[:, 2], 'b*', markersize=12, label='Predictions')
# Legend and plot
# plt.legend(loc='best', fontsize=10)
plt.legend(loc='best', fontsize=24)
plt.xlabel('Samples')
plt.ylabel('Tortuosity')
# plt.legend(loc='best', fontsize=10)
# plt.legend(loc='best', fontsize=15)
plt.savefig('Tortuosity_TestData.png', bbox_inches='tight')

plt.figure()
# plt.subplot(235)
xx = scaler2.inverse_transform(y_train)[:, 2]
yy = scaler2.inverse_transform(predicted_on_X_train)[:, 2]
print('JCA - tortuosity  - r2_score')
print(r2_score(xx, yy))  #r2_score(y_true, y_pred)
print('JCA - tortuosity  - mean_squared_error')
print(mean_squared_error(xx, yy))  #mean_squared_error(y_true, y_pred)
print('JCA - tortuosity  - mean_absolute_error')
print(mean_absolute_error(xx, yy))  #mean_absolute_error(y_true, y_pred)
print('JCA - tortuosity  - mean_absolute_percentage_error')
print(mean_absolute_percentage_error(xx, yy))  #mean_absolute_percentage_error(y_true, y_pred)

x_major_locator=MultipleLocator(0.3)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.3)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
xx_validation = scaler2.inverse_transform(y_validation)[:, 2]
yy_validation = scaler2.inverse_transform(predicted_on_X_validation)[:, 2]
xx_test = scaler2.inverse_transform(y_test)[:, 2]
yy_test = scaler2.inverse_transform(predicted_on_X_test)[:, 2]

print('JCA - tortuosity_validation - mean_absolute_percentage_error')
print(mean_absolute_percentage_error(xx_validation, yy_validation))  #mean_absolute_percentage_error(y_true, y_pred)
print('JCA - tortuosity_test - mean_absolute_percentage_error')
print(mean_absolute_percentage_error(xx_test, yy_test))  #mean_absolute_percentage_error(y_true, y_pred)

bubble_plot_line_x1y1 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
bubble_plot_line_x2y2 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx, yy, label='Tortuosity', marker='o', facecolors='none', edgecolors='red', s=140)
# plt.legend(loc='best')
plt.title('Tortuosity')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Tortuosity_Training.png', bbox_inches='tight')

plt.figure()
x_major_locator=MultipleLocator(0.3)
y_major_locator=MultipleLocator(0.3)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
bubble_plot_line_x1y1 = [min(np.minimum(xx_validation, yy_validation)), max(np.maximum(xx_validation, yy_validation))]
bubble_plot_line_x2y2 = [min(np.minimum(xx_validation, yy_validation)), max(np.maximum(xx_validation, yy_validation))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx_validation, yy_validation, label='Tortuosity', marker='o', facecolors='none', edgecolors='blue', s=140)
# plt.legend(loc='best')
plt.title('Tortuosity')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Tortuosity_Validation.png', bbox_inches='tight')

plt.figure()
x_major_locator=MultipleLocator(0.4)
y_major_locator=MultipleLocator(0.4)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
bubble_plot_line_x1y1 = [min(np.minimum(xx_test, yy_test)), max(np.maximum(xx_test, yy_test))]
bubble_plot_line_x2y2 = [min(np.minimum(xx_test, yy_test)), max(np.maximum(xx_test, yy_test))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx_test, yy_test, label='Tortuosity', marker='o', facecolors='none', edgecolors='blue', s=140)
# plt.legend(loc='best')
plt.title('Tortuosity')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Tortuosity_Test.png', bbox_inches='tight')

# print()
# print("o/p of test set:           \n", (scaler2.inverse_transform(y_test)[:, 2]))
# print("predicted o/p of test set: \n", (scaler2.inverse_transform(predicted_on_X_test)[:, 2]))
# print("mse_test_set: ", mean_squared_error(y_test, predicted_on_X_test))


###############################################################
#################   plotting graphs together - viscous length  ################
###############################################################

plt.figure()
# plt.suptitle('JCA - viscous length - (epochs-{}) - pyTorch'.format(epochs), fontsize=25,
#              color='r', fontweight='bold')  ## giving title on top of all subplots

# plt.subplot(231)
plt.plot(running_loss_index, running_loss, 'r-', linewidth=3, label='MSE_loss_train')
plt.plot(running_loss_validation_index, running_loss_validation, 'b-', linewidth=3, label='MSE_loss_validation')
# plt.legend(loc='best', fontsize=10)
# plt.xlabel('epochs#', fontsize=15)
plt.legend()
plt.ylabel('MSE loss')
plt.xlabel('Epochs')
plt.savefig('Characteristic viscous length_MES_Epochs.png', bbox_inches='tight')

plt.figure()
# plt.subplot(234)
# Plot true data
plt.plot(scaler2.inverse_transform(y_test)[:, 3], 'ro', markersize=12, label='Targets')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_test)[:, 3], 'b*', markersize=12, label='Predictions')
# Legend and plot
# plt.legend(loc='best', fontsize=10)
ax=plt.gca()
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='y') #  x y or both
plt.xlabel('Samples')
plt.ylabel('Characteristic viscous length')
# plt.legend(loc='best', fontsize=10)
# plt.legend(loc='best', fontsize=15)
plt.legend(loc='best', fontsize=24)
plt.savefig('Characteristic viscous length_TestData.png', bbox_inches='tight')

plt.figure()
# plt.subplot(235)
xx = scaler2.inverse_transform(y_train)[:, 3]
yy = scaler2.inverse_transform(predicted_on_X_train)[:, 3]
print('JCA - viscous length  - r2_score')
print(r2_score(xx, yy))  #r2_score(y_true, y_pred)
print('JCA - viscous length  - mean_squared_error')
print(mean_squared_error(xx, yy))  #mean_squared_error(y_true, y_pred)
print('JCA - viscous length  - mean_absolute_error')
print(mean_absolute_error(xx, yy))  #mean_absolute_error(y_true, y_pred)
print('JCA - viscous length  - mean_absolute_percentage_error')
print(mean_absolute_percentage_error(xx, yy))  #mean_absolute_percentage_error(y_true, y_pred)

plt.figure()
x_major_locator=MultipleLocator(0.0005)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.0005)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='both') #  x y or both
xx_validation = scaler2.inverse_transform(y_validation)[:, 3]
yy_validation = scaler2.inverse_transform(predicted_on_X_validation)[:, 3]
xx_test = scaler2.inverse_transform(y_test)[:, 3]
yy_test = scaler2.inverse_transform(predicted_on_X_test)[:, 3]

print('JCA - viscous length_validation - mean_absolute_percentage_error')
print(mean_absolute_percentage_error(xx_validation, yy_validation))  #mean_absolute_percentage_error(y_true, y_pred)
print('JCA - viscous length_test - mean_absolute_percentage_error')
print(mean_absolute_percentage_error(xx_test, yy_test))  #mean_absolute_percentage_error(y_true, y_pred)

bubble_plot_line_x1y1 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
bubble_plot_line_x2y2 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx, yy, label='Characteristic viscous length', marker='o', facecolors='none', edgecolors='red', s=140)
# plt.legend(loc='best', fontsize=20)
plt.title('Characteristic viscous length')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Characteristic viscous length_Training.png', bbox_inches='tight')

plt.figure()
x_major_locator=MultipleLocator(0.0005)
y_major_locator=MultipleLocator(0.0005)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='both') #  x y or both
bubble_plot_line_x1y1 = [min(np.minimum(xx_validation, yy_validation)), max(np.maximum(xx_validation, yy_validation))]
bubble_plot_line_x2y2 = [min(np.minimum(xx_validation, yy_validation)), max(np.maximum(xx_validation, yy_validation))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx_validation, yy_validation, label='Characteristic viscous length', marker='o', facecolors='none', edgecolors='blue', s=140)
# plt.legend(loc='best', fontsize=20)
plt.title('Characteristic viscous length')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Characteristic viscous length_Validation.png', bbox_inches='tight')

plt.figure()
x_major_locator=MultipleLocator(0.0002)
y_major_locator=MultipleLocator(0.0002)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='both') #  x y or both
bubble_plot_line_x1y1 = [min(np.minimum(xx_test, yy_test)), max(np.maximum(xx_test, yy_test))]
bubble_plot_line_x2y2 = [min(np.minimum(xx_test, yy_test)), max(np.maximum(xx_test, yy_test))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx_test, yy_test, label='Characteristic viscous length', marker='o', facecolors='none', edgecolors='blue', s=140)
# plt.legend(loc='best', fontsize=20)
plt.title('Characteristic viscous length')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Characteristic viscous length_Test.png', bbox_inches='tight')

plt.figure()
x_major_locator=MultipleLocator(0.0002)
y_major_locator=MultipleLocator(0.0002)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='both') #  x y or both
bubble_plot_line_x1y1 = [min(np.minimum(xx_test, yy_test)), max(np.maximum(xx_test, yy_test))]
bubble_plot_line_x2y2 = [min(np.minimum(xx_test, yy_test)), max(np.maximum(xx_test, yy_test))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx_test, yy_test, label='Characteristic viscous length', marker='o', facecolors='none', edgecolors='blue', s=140)
# plt.legend(loc='best', fontsize=20)
plt.title('Characteristic viscous length')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Characteristic viscous length_Test.png', bbox_inches='tight')

# plt.figure()
# # plt.subplot(236)
# true_values = scaler2.inverse_transform(y_test)[:, 3]
# predicted_values = scaler2.inverse_transform(predicted_on_X_test)[:, 3]
# x_index = [i for i in range(len(true_values))]
# error_values = predicted_values - true_values
# plt.errorbar(x=x_index, y=true_values, yerr=error_values, fmt='o', color='black',
#              ecolor='black', elinewidth=2, capsize=10);
# plt.grid(linestyle='--', linewidth=1)

# print()
# print("o/p of test set:           \n", (scaler2.inverse_transform(y_test)[:, 3]))
# print("predicted o/p of test set: \n", (scaler2.inverse_transform(predicted_on_X_test)[:, 3]))
# print("mse_test_set: ", mean_squared_error(y_test, predicted_on_X_test))

plt.figure()
# plt.suptitle('JCA - thermal length - (epochs-{}) - pyTorch'.format(epochs), fontsize=25,
#              color='r', fontweight='bold')  ## giving title on top of all subplots

# plt.subplot(231)
plt.plot(running_loss_index, running_loss, 'r-', linewidth=3, label='MSE_loss_train')
plt.plot(running_loss_validation_index, running_loss_validation, 'b-', linewidth=3, label='MSE_loss_validation')
# plt.legend(loc='best', fontsize=10)
plt.legend()
plt.ylabel('MSE loss')
plt.xlabel('Epochs')
plt.savefig('Characteristic thermal length_MES_Epochs.png', bbox_inches='tight')

plt.figure()
# plt.subplot(234)
# Plot true data
plt.plot(scaler2.inverse_transform(y_test)[:, 4], 'ro', markersize=12, label='Targets')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_test)[:, 4], 'b*', markersize=12, label='Predictions')
# Legend and plot
# plt.legend(loc='best', fontsize=10)
plt.xlabel('Samples')
plt.ylabel('Characteristic thermal length')
plt.legend(loc='best', fontsize=24)
ax=plt.gca()
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='y') #  x y or both
# plt.legend(loc='best', fontsize=10)
# plt.legend(loc='best', fontsize=15)
plt.savefig('Characteristic thermal length_TestData.png', bbox_inches='tight')

plt.figure()
# plt.subplot(235)
xx = scaler2.inverse_transform(y_train)[:, 4]
yy = scaler2.inverse_transform(predicted_on_X_train)[:, 4]
print('JCA - thermal length  - r2_score')
print(r2_score(xx, yy))  #r2_score(y_true, y_pred)
print('JCA - thermal length  - mean_squared_error')
print(mean_squared_error(xx, yy))  #mean_squared_error(y_true, y_pred)
print('JCA - thermal length  - mean_absolute_error')
print(mean_absolute_error(xx, yy))  #mean_absolute_error(y_true, y_pred)
print('JCA - thermal length  - mean_absolute_percentage_error')
print(mean_absolute_percentage_error(xx, yy))  #mean_absolute_percentage_error(y_true, y_pred)

x_major_locator=MultipleLocator(0.0008)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.0008)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='both') #  x y or both

xx_validation = scaler2.inverse_transform(y_validation)[:, 4]
yy_validation = scaler2.inverse_transform(predicted_on_X_validation)[:, 4]
xx_test = scaler2.inverse_transform(y_test)[:, 4]
yy_test = scaler2.inverse_transform(predicted_on_X_test)[:, 4]

print('JCA - thermal length_validation - mean_absolute_percentage_error')
print(mean_absolute_percentage_error(xx_validation, yy_validation))  #mean_absolute_percentage_error(y_true, y_pred)
print('JCA - thermal length_test - mean_absolute_percentage_error')
print(mean_absolute_percentage_error(xx_test, yy_test))  #mean_absolute_percentage_error(y_true, y_pred)

bubble_plot_line_x1y1 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
bubble_plot_line_x2y2 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx, yy, label='Characteristic thermal length', marker='o', facecolors='none', edgecolors='red', s=140)
# plt.legend(loc='best')
# plt.legend(loc='best', fontsize=20)
plt.title('Characteristic thermal length')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Characteristic thermal length_Training.png', bbox_inches='tight')

plt.figure()
x_major_locator=MultipleLocator(0.0008)
y_major_locator=MultipleLocator(0.0008)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='both') #  x y or both
bubble_plot_line_x1y1 = [min(np.minimum(xx_validation, yy_validation)), max(np.maximum(xx_validation, yy_validation))]
bubble_plot_line_x2y2 = [min(np.minimum(xx_validation, yy_validation)), max(np.maximum(xx_validation, yy_validation))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx_validation, yy_validation, label='Characteristic thermal length', marker='o', facecolors='none', edgecolors='blue', s=140)
# plt.legend(loc='best', fontsize=20)
plt.title('Characteristic thermal length')
# title = plt.title(title="Characteristic thermal length")
# title._legend_box.align = "left"
# legend._legend_box.align = "left"/"right"/"center"
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Characteristic thermal length_Validation.png', bbox_inches='tight')


plt.figure()
x_major_locator=MultipleLocator(0.0008)
y_major_locator=MultipleLocator(0.0008)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='both') #  x y or both
bubble_plot_line_x1y1 = [min(np.minimum(xx_test, yy_test)), max(np.maximum(xx_test, yy_test))]
bubble_plot_line_x2y2 = [min(np.minimum(xx_test, yy_test)), max(np.maximum(xx_test, yy_test))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
# plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx_test, yy_test, label='Characteristic thermal length', marker='o', facecolors='none', edgecolors='blue', s=140)
# plt.legend(loc='best', fontsize=20)
plt.title('Characteristic thermal length')
# title = plt.title(title="Characteristic thermal length")
# title._legend_box.align = "left"
# legend._legend_box.align = "left"/"right"/"center"
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('Characteristic thermal length_Test.png', bbox_inches='tight')

# print()
# print("o/p of test set:           \n", (scaler2.inverse_transform(y_test)[:, 4]))
# print("predicted o/p of test set: \n", (scaler2.inverse_transform(predicted_on_X_test)[:, 4]))
# print("mse_test_set: ", mean_squared_error(y_test, predicted_on_X_test))
# print()

# ###########    saving predicted data to excel file   ##############
# l1 = 10 ** (scaler2.inverse_transform(y_test)[:, 4])  #### check multiple of 10
# l2 = 10 ** (scaler2.inverse_transform(predicted_on_X_test)[:, 4])  #### check multiple of 10
# ###########    saving mse data to excel file   ##############
# l3 = running_loss
# l4 = running_loss_validation
# l5 = running_loss_index
# l6 = running_loss_validation_index
#
# ## convert your array into a dataframe
# # df = pd.DataFrame(l1, columns=['a'])
# df = pd.DataFrame({'a': l3, 'b': l4})
# ## save to xlsx file
# filepath = 'test_excel_file.xlsx'
# df.to_excel(filepath, sheet_name='sheet1', index=False)

# sys.exit()

f.write('\n')
xx = scaler2.inverse_transform(y_train)[:, 0]
yy = scaler2.inverse_transform(predicted_on_X_train)[:, 0]
xx_validation = scaler2.inverse_transform(y_validation)[:, 0]
yy_validation = scaler2.inverse_transform(predicted_on_X_validation)[:, 0]
xx_test = scaler2.inverse_transform(y_test)[:, 0]
yy_test = scaler2.inverse_transform(predicted_on_X_test)[:, 0]


f.write("JCA - Training porosity - r2_score")
f.write('\n')
f.write(str(r2_score(xx, yy)))
f.write('\n')
f.write("JCA - Training porosity - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx, yy)))
f.write('\n')
f.write("JCA - Training porosity - mean_squared_error_ -LOG converge_speed")
f.write('\n')
f.write(str(-math.log10(mean_squared_error(xx, yy))))
f.write('\n')
f.write("JCA - Training porosity - root mean_squared_error")  # mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx, yy))))
f.write('\n')
f.write("JCA - Training porosity - mean_absolute_error")  # mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx, yy)))
f.write('\n')
f.write("JCA - Training porosity - mean_absolute_percentage_error")
f.write('\n')
f.write(str(mean_absolute_percentage_error(xx, yy)))
f.write('\n')
f.write("JCA - Validation porosity - r2_score")
f.write('\n')
f.write(str(r2_score(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation porosity - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation porosity - -LOG mean_squared_error")
f.write('\n')
f.write(str(-math.log10(mean_squared_error(xx, yy))))
f.write('\n')
f.write("JCA - Validation porosity - root mean_squared_error")  # mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx_validation, yy_validation))))
f.write('\n')
f.write("JCA - Validation porosity - mean_absolute_error")  # mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation porosity - mean_absolute_percentage_error")
f.write('\n')
f.write(str(mean_absolute_percentage_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Testing porosity - r2_score")
f.write('\n')
f.write(str(r2_score(xx_test, yy_test)))
f.write('\n')
f.write("JCA - Testing porosity - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_test, yy_test)))
f.write('\n')
f.write("JCA - Testing porosity - mean_absolute_percentage_error")
f.write('\n')
f.write(str(mean_absolute_percentage_error(xx_test, yy_test)))

f.write('\n')
xx = scaler2.inverse_transform(y_train)[:, 1]
yy = scaler2.inverse_transform(predicted_on_X_train)[:, 1]
xx_validation = scaler2.inverse_transform(y_validation)[:, 1]
yy_validation = scaler2.inverse_transform(predicted_on_X_validation)[:, 1]
xx_test = scaler2.inverse_transform(y_test)[:, 1]
yy_test = scaler2.inverse_transform(predicted_on_X_test)[:, 1]

f.write('\n')
f.write("JCA - Training flow resistance - r2_score")
f.write('\n')
f.write(str(r2_score(xx, yy)))
f.write('\n')
f.write("JCA - Training flow resistance - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx, yy)))
f.write('\n')
f.write("JCA - Training flow resistance - mean_squared_error_ converge_speed")
f.write('\n')
f.write(str(-math.log10(mean_squared_error(xx, yy))))
f.write('\n')
f.write("JCA - Training flow resistance - root mean_squared_error")  # mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx, yy))))
f.write('\n')
f.write("JCA - Training flow resistance - mean_absolute_error")  # mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx, yy)))
f.write('\n')
f.write("JCA - Training flow resistance - mean_absolute_percentage_error")
f.write('\n')
f.write(str(mean_absolute_percentage_error(xx, yy)))

f.write('\n')
f.write("JCA - Validation flow resistance - r2_score")
f.write('\n')
f.write(str(r2_score(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation flow resistance - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation flow resistance - -LOG mean_squared_error")
f.write('\n')
f.write(str(-math.log10(mean_squared_error(xx_validation, yy_validation))))
f.write('\n')
f.write("JCA - Validation flow resistance - root mean_squared_error")  # mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx_validation, yy_validation))))
f.write('\n')
f.write("JCA - Validation flow resistance - mean_absolute_error")  # mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation flow resistance - mean_absolute_percentage_error")
f.write('\n')
f.write(str(mean_absolute_percentage_error(xx_validation, yy_validation)))



f.write('\n')
f.write("JCA - Testing flow resistance - r2_score")
f.write('\n')
f.write(str(r2_score(xx_test, yy_test)))
f.write('\n')
f.write("JCA - Testing flow resistance - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_test, yy_test)))
f.write('\n')
f.write("JCA - Testing flow resistance - mean_absolute_percentage_error")
f.write('\n')
f.write(str(mean_absolute_percentage_error(xx_test, yy_test)))

print("JCA - Validation flow resistance - r2_score")
print(str(r2_score(xx_validation, yy_validation)))

print('JCA - Testing flow resistance - r2_score')
print(r2_score(xx_test, yy_test))  #r2_score(y_true, y_pred)

f.write('\n')
xx = scaler2.inverse_transform(y_train)[:, 2]
yy = scaler2.inverse_transform(predicted_on_X_train)[:, 2]
xx_validation = scaler2.inverse_transform(y_validation)[:, 2]
yy_validation = scaler2.inverse_transform(predicted_on_X_validation)[:, 2]
xx_test = scaler2.inverse_transform(y_test)[:, 2]
yy_test = scaler2.inverse_transform(predicted_on_X_test)[:, 2]

f.write('\n')
f.write("JCA - Training  tortuosity - r2_score")
f.write('\n')
f.write(str(r2_score(xx, yy)))
f.write('\n')
f.write("JCA - Training  tortuosity - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx, yy)))
f.write('\n')
f.write("JCA - Training  tortuosity - root mean_squared_error")  # mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx, yy))))
f.write('\n')
f.write("JCA - Training  tortuosity - mean_absolute_error")  # mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx, yy)))
f.write('\n')
f.write("JCA - Training tortuosity - mean_absolute_percentage_error")
f.write('\n')
f.write(str(mean_absolute_percentage_error(xx, yy)))

f.write('\n')
f.write("JCA - Validation  tortuosity - r2_score")
f.write('\n')
f.write(str(r2_score(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation  tortuosity - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation  tortuosity - -LOG mean_squared_error")
f.write('\n')
f.write(str(-math.log10(mean_squared_error(xx_validation, yy_validation))))
f.write('\n')
f.write("JCA - Validation  tortuosity - root mean_squared_error")  # mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx_validation, yy_validation))))
f.write('\n')
f.write("JCA - Validation  tortuosity - mean_absolute_error")  # mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation tortuosity - mean_absolute_percentage_error")
f.write('\n')
f.write(str(mean_absolute_percentage_error(xx_validation, yy_validation)))


f.write('\n')
f.write("JCA - Testing tortuosity - r2_score")
f.write('\n')
f.write(str(r2_score(xx_test, yy_test)))
f.write('\n')
f.write("JCA - Testing tortuosity - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_test, yy_test)))
f.write('\n')
f.write("JCA - Testing tortuosity - mean_absolute_percentage_error")
f.write('\n')
f.write(str(mean_absolute_percentage_error(xx_test, yy_test)))

print("JCA - Validation tortuosity - r2_score")
print(str(r2_score(xx_validation, yy_validation)))

print('JCA - Testing tortuosity- r2_score')
print(r2_score(xx_test, yy_test))  #r2_score(y_true, y_pred)


f.write('\n')
xx = scaler2.inverse_transform(y_train)[:, 3]
yy = scaler2.inverse_transform(predicted_on_X_train)[:, 3]
xx_validation = scaler2.inverse_transform(y_validation)[:, 3]
yy_validation = scaler2.inverse_transform(predicted_on_X_validation)[:, 3]
xx_test = scaler2.inverse_transform(y_test)[:, 3]
yy_test = scaler2.inverse_transform(predicted_on_X_test)[:, 3]

f.write('\n')
f.write('\n')
f.write("JCA - Training viscous length - r2_score")
f.write('\n')
f.write(str(r2_score(xx, yy)))
f.write('\n')
f.write("JCA - Training viscous length - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx, yy)))
f.write('\n')
f.write("JCA - Training viscous length - root mean_squared_error")  # mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx, yy))))
f.write('\n')
f.write("JCA - Training viscous length - mean_absolute_error")  # mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx, yy)))
f.write('\n')
f.write("JCA - Training viscous length  - mean_absolute_percentage_error")
f.write('\n')
f.write(str(mean_absolute_percentage_error(xx, yy)))

f.write('\n')
f.write("JCA - Validation viscous length - r2_score")
f.write('\n')
f.write(str(r2_score(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation viscous length - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation viscous length - -LOG mean_squared_error")
f.write('\n')
f.write(str(-math.log10(mean_squared_error(xx_validation, yy_validation))))
f.write('\n')
f.write("JCA - Validation viscous length - root mean_squared_error")  # mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx_validation, yy_validation))))
f.write('\n')
f.write("JCA - Validation viscous length - mean_absolute_error")  # mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation viscous length  - mean_absolute_percentage_error")
f.write('\n')
f.write(str(mean_absolute_percentage_error(xx_validation, yy_validation)))


f.write('\n')
f.write("JCA - Testing viscous length - r2_score")
f.write('\n')
f.write(str(r2_score(xx_test, yy_test)))
f.write('\n')
f.write("JCA - Testing viscous length - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_test, yy_test)))
f.write('\n')
f.write("JCA - Testing viscous length  - mean_absolute_percentage_error")
f.write('\n')
f.write(str(mean_absolute_percentage_error(xx_test, yy_test)))

f.write('\n')
xx = scaler2.inverse_transform(y_train)[:, 4]
yy = scaler2.inverse_transform(predicted_on_X_train)[:, 4]
xx_validation = scaler2.inverse_transform(y_validation)[:, 4]
yy_validation = scaler2.inverse_transform(predicted_on_X_validation)[:, 4]
xx_test = scaler2.inverse_transform(y_test)[:, 4]
yy_test = scaler2.inverse_transform(predicted_on_X_test)[:, 4]

f.write('\n')
f.write("JCA - Training thermal length - r2_score")
f.write('\n')
f.write(str(r2_score(xx, yy)))
f.write('\n')
f.write("JCA - Training thermal length - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx, yy)))
f.write('\n')
f.write("JCA - Training thermal length - root mean_squared_error")  # mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx, yy))))
f.write('\n')
f.write("JCA - Training thermal length - mean_absolute_error")  # mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx, yy)))
f.write('\n')
f.write("JCA - Training thermal length  - mean_absolute_percentage_error")
f.write('\n')
f.write(str(mean_absolute_percentage_error(xx, yy)))

f.write('\n')
f.write("JCA - Validation thermal length - r2_score")
f.write('\n')
f.write(str(r2_score(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation thermal length - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation thermal length - -LOG mean_squared_error")
f.write('\n')
f.write(str(-math.log10(mean_squared_error(xx_validation, yy_validation))))
f.write('\n')
f.write("JCA - Validation thermal length - root mean_squared_error")  # mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx_validation, yy_validation))))
f.write('\n')
f.write("JCA - Validation thermal length - mean_absolute_error")  # mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation thermal length  - mean_absolute_percentage_error")
f.write('\n')
f.write(str(mean_absolute_percentage_error(xx_validation, yy_validation)))

f.write('\n')
f.write("JCA - Testing thermal length - r2_score")
f.write('\n')
f.write(str(r2_score(xx_test, yy_test)))
f.write('\n')
f.write("JCA - Testing thermal length - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_test, yy_test)))
f.write('\n')
f.write("JCA - Testing thermal length  - mean_absolute_percentage_error")
f.write('\n')
f.write(str(mean_absolute_percentage_error(xx_test, yy_test)))

f.close()

print("JCA - Validation thermal length - r2_score")
print(str(r2_score(xx_validation, yy_validation)))

print('JCA - Testing thermal length - r2_score')
print(r2_score(xx_test, yy_test))  #r2_score(y_true, y_pred)

# plt.show()

##########################################
#######           GUI           ##########
##########################################

