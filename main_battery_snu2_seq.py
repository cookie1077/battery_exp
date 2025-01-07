import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
import os
import csv
import warnings

from model import MLP, RNN_LSTM, RNN_GRU, TransformerModel

device = "cuda" if torch.cuda.is_available() else "cpu"

warnings.simplefilter(action='ignore', category=FutureWarning)

"""
Experiments using the single feature for thesis
"""
####################################################################################
seq_length = 3
####################################################################################
data_path = "dataset_snu2_240826/"
model_path = f"Model_Thesis/model_snu2_Selected_13Features_{seq_length}Seq/"
result_path = f"Results_Thesis/results_snu2_Selected_13Features_{seq_length}Seq/"


try:
    if not os.path.exists(model_path):
        os.makedirs(model_path)
except OSError:
    print('Error: Creating directory. ' + model_path)


try:
    if not os.path.exists(result_path):
        os.makedirs(result_path)
except OSError:
    print('Error: Creating directory. ' + result_path)

data_list = os.listdir(data_path)

"""
논문에 사용할 13개 Features
num_selected = [2, 1, 146, 57, 141, 68, 25, 99, 95, 87, 78, 108, 152]
"""
norm_factors = 1
rul_factor = 800  # Normalization factor for RUL
####################################################################################
all_results_mlp = []
all_results_gru = []
all_results_lstm = []
all_results_transformer = []

csv_filename_mlp = os.path.join(result_path, "results_mlp.csv")
csv_filename_gru = os.path.join(result_path, "results_gru.csv")
csv_filename_lstm = os.path.join(result_path, "results_lstm.csv")
csv_filename_transformer = os.path.join(result_path, "results_transformer.csv")
####################################################################################


def build_dataset(data_x, data_y, seq_length):
    data_out_x = []
    data_out_y = []

    for i in range(0, len(data_x) - seq_length):
        _x = data_x[i:i + seq_length, :]
        data_out_x.append(_x)

    for i in range(0, len(data_y) - seq_length):
        _y = data_y[i+seq_length, :]
        data_out_y.append(_y)

    data_out_x = np.array(data_out_x)
    data_out_x = torch.FloatTensor(data_out_x)
    data_out_x = data_out_x.to(device)

    data_out_y = np.array(data_out_y)
    data_out_y = torch.FloatTensor(data_out_y)
    data_out_y = data_out_y.to(device)

    return data_out_x, data_out_y


def generate_randomness(max_num_features, num_features):
    seed = random.randint(1, 1000)
    num_selected = random.sample(range(0, max_num_features), num_features)

    return num_selected, seed


def run_experiment_mlp(iteration, num_selected, seed, seq_length):
    ####################
    epochs = 1000
    train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=seed)

    loss_train_list = []
    loss_test_list = []
    ####################
    print(f"[MLP] Experiment {iteration} started.")

    results_filename = os.path.join(result_path, f"results_MLP.txt")

    # iteration_path = os.path.join(result_path, f"MLP_iter_{iteration}") # Create a directory for the current iteration
    # os.makedirs(iteration_path, exist_ok=True)

    rmse_scores = []
    r2_scores = []
    rmse_dict = {}  # To store RMSE for each data_name
    r2_dict = {}    # To store R2 for each data_name

    # train
    model = MLP(len(num_selected), seq_length, 64, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    for epoch in range(epochs):
        random.shuffle(train_list)
        epoch_loss = 0.0

        for data_name in train_list:
            data = torch.load(data_path + data_name)

            x = data[:, num_selected] / norm_factors
            y = data[:, -1][:, None] / rul_factor  # RUL

            train_x, train_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)

            loss = model.loss(train_x, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu().numpy()

        scheduler.step()

        if epoch % 10 == 9:  # Evaluating the test set for every 10 epochs
            rmse_scores_train_tmp = []
            rmse_scores_test_tmp = []
            r2_scores_test_tmp = []

            ############## Train Files ##############
            for data_name in train_list:
                data = torch.load(data_path + data_name)

                x = data[:, num_selected] / norm_factors
                y = data[:, -1][:, None]  # RUL

                train_x, train_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
                y_true_tmp, y_pred_tmp = [], []

                with torch.no_grad():
                    cycle = data[seq_length:, 0]

                    y_ = model(train_x) * rul_factor

                    y_true_tmp.append(train_y)
                    y_pred_tmp.append(y_)
                    y_true_tmp = torch.cat(y_true_tmp, axis=0)
                    y_pred_tmp = torch.cat(y_pred_tmp, axis=0)

                    rmse_loss_tmp = np.sqrt(
                        mean_squared_error(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy()))
                    rmse_scores_train_tmp.append(rmse_loss_tmp)

            average_rmse_train_tmp = np.mean(rmse_scores_train_tmp)

            loss_train_list.append(average_rmse_train_tmp)

            ############## Test Files ##############
            for file_name in test_list:
                test_data = torch.load(data_path + file_name)

                x = test_data[:, num_selected] / norm_factors
                y = test_data[:, -1][:, None]  # RUL

                test_x, test_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
                y_true_tmp, y_pred_tmp = [], []

                with torch.no_grad():
                    cycle = test_data[seq_length:, 0]

                    y_ = model(test_x) * rul_factor

                    y_true_tmp.append(test_y)
                    y_pred_tmp.append(y_)
                    y_true_tmp = torch.cat(y_true_tmp, axis=0)
                    y_pred_tmp = torch.cat(y_pred_tmp, axis=0)

                    rmse_loss_tmp = np.sqrt(mean_squared_error(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy()))
                    r2_tmp = r2_score(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy())
                    if not np.isnan(rmse_loss_tmp) and not np.isnan(r2_tmp):
                        rmse_scores_test_tmp.append(rmse_loss_tmp)
                        r2_scores_test_tmp.append(r2_tmp)

            if rmse_scores_test_tmp and r2_scores_test_tmp:
                average_rmse_test_tmp = np.mean(rmse_scores_test_tmp)
                average_r2_test_tmp = np.mean(r2_scores_test_tmp)
            else:
                average_rmse_test_tmp = np.nan
                average_r2_test_tmp = np.nan
            print(f"Epoch: {epoch}, Average RMSE: {average_rmse_test_tmp}, Average R2: {average_r2_test_tmp}")

            loss_test_list.append(average_rmse_test_tmp)

    # plt.plot(loss_train_list, c="blue", label="Train loss")
    # plt.plot(loss_test_list, c="red", label="Test Loss")
    # plt.title(f"Train and Test Loss")
    # plt.xlabel("10 Epochs")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig(os.path.join(iteration_path, f"Train and Test Loss.png"), dpi=200)
    # plt.close()

    model_name = f"mlp_{iteration}"
    model_save_path = model_path + f"{iteration}/"

    try:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
    except OSError:
        print('Error: Creating directory. ' + model_save_path)

    torch.save(model, model_save_path + model_name)

    # # Evaluating the model
    # model = torch.load(model_path + model_name)

    for file_name in test_list:
        test_data = torch.load(data_path + file_name)

        x = test_data[:, num_selected] / norm_factors
        y = test_data[:, -1][:, None]  # RUL

        test_x, test_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            cycle = test_data[seq_length:, 0]

            y_ = model(test_x) * rul_factor

            y_true.append(test_y)
            y_pred.append(y_)
            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            rmse_loss = np.sqrt(mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()))
            r2 = r2_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

            rmse_scores.append(rmse_loss)
            r2_scores.append(r2)
            rmse_dict[file_name] = rmse_loss  # Store RMSE with its corresponding data_name
            r2_dict[file_name] = r2

        # plt.plot(y_true.detach().cpu().numpy(), y_true.detach().cpu().numpy(), c="red", label="True RUL")
        # plt.plot(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), c="blue", label="Predicted RUL")
        # plt.title(f"RUL Prediction of the Battery Cell #{file_name[-4:]}")
        # plt.xlabel("True RUL")
        # plt.ylabel("Predicted RUL")
        # plt.legend()
        # plt.savefig(os.path.join(iteration_path, f"Cell #{file_name[-4:]}.png"), dpi=200)
        # plt.close()

        with open(results_filename, "a") as file:
            file.write(f"Iteration {iteration} // {file_name}\n")
            file.write(f"RMSE: {rmse_loss}\n")
            file.write(f"R2: {r2}\n\n")

    average_rmse = np.mean(rmse_scores)
    print(average_rmse)
    std_dev_rmse = np.std(rmse_scores)
    average_r2 = np.mean(r2_scores)
    std_dev_r2 = np.std(r2_scores)
    max_rmse_data_name = max(rmse_dict, key=rmse_dict.get)
    min_r2_data_name = min(r2_dict, key=r2_dict.get)

    with open(results_filename, "a") as file:
        file.write(f"Iteration {iteration}, Selected Features: {num_selected}, Split_Seed: {seed},"
                   f" Epochs: {epochs}, Sequence Length: {seq_length}\n")
        file.write(f"Average RMSE: {average_rmse}, Std Dev: {std_dev_rmse}\n")
        file.write(f"Average R2: {average_r2}, Std Dev: {std_dev_r2}\n")
        file.write(f"Data Name with Highest RMSE: {max_rmse_data_name}, RMSE: {rmse_dict[max_rmse_data_name]}\n")
        file.write(f"Data Name with Smallest R2: {min_r2_data_name}, R2: {r2_dict[min_r2_data_name]}\n\n")
        file.write("=============================================================================\n\n")

    all_results_mlp.append({
        # "Feature Index": num_selected,
        "Iteration": iteration,
        "Average RMSE": average_rmse,
        "RMSE Std Dev": std_dev_rmse,
        "Average R2": average_r2,
        "R2 Std Dev": std_dev_r2,
        "Max RMSE Data Name": max_rmse_data_name,
        "Max RMSE Value": rmse_dict[max_rmse_data_name],
        "Min R2 Data Name": min_r2_data_name,
        "Min R2 Value": r2_dict[min_r2_data_name]
    })
    print(f"[MLP] Experiment {iteration} completed and results saved.")
    print("")


def run_experiment_gru(iteration, num_selected, seed, seq_length):
    ####################
    epochs = 1000
    train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=seed)

    loss_train_list = []
    loss_test_list = []
    ####################
    print(f"[GRU] Experiment {iteration} started.")

    results_filename = os.path.join(result_path, f"results_gru.txt")

    # iteration_path = os.path.join(result_path, f"GRU_iter_{iteration}") # Create a directory for the current iteration
    # os.makedirs(iteration_path, exist_ok=True)

    rmse_scores = []
    r2_scores = []
    rmse_dict = {}  # To store RMSE for each data_name
    r2_dict = {}    # To store R2 for each data_name

    # train
    model = RNN_GRU(input_dim=len(num_selected), hidden_dim=10, output_dim=1, layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    for epoch in range(epochs):
        random.shuffle(train_list)
        epoch_loss = 0.0

        for data_name in train_list:
            data = torch.load(data_path + data_name)

            x = data[:, num_selected] / norm_factors
            y = data[:, -1][:, None] / rul_factor  # RUL

            train_x, train_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)

            loss = model.loss(train_x, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu().numpy()

        scheduler.step()

        if epoch % 10 == 9:  # Evaluating the test set for every 10 epochs
            rmse_scores_train_tmp = []
            rmse_scores_test_tmp = []
            r2_scores_test_tmp = []

            ############## Train Files ##############
            for data_name in train_list:
                data = torch.load(data_path + data_name)

                x = data[:, num_selected] / norm_factors
                y = data[:, -1][:, None]  # RUL

                train_x, train_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
                y_true_tmp, y_pred_tmp = [], []

                with torch.no_grad():
                    cycle = data[seq_length:, 0]

                    y_ = model(train_x) * rul_factor

                    y_true_tmp.append(train_y)
                    y_pred_tmp.append(y_)
                    y_true_tmp = torch.cat(y_true_tmp, axis=0)
                    y_pred_tmp = torch.cat(y_pred_tmp, axis=0)

                    rmse_loss_tmp = np.sqrt(
                        mean_squared_error(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy()))
                    rmse_scores_train_tmp.append(rmse_loss_tmp)

            average_rmse_train_tmp = np.mean(rmse_scores_train_tmp)

            loss_train_list.append(average_rmse_train_tmp)

            ############## Test Files ##############
            for file_name in test_list:
                test_data = torch.load(data_path + file_name)

                x = test_data[:, num_selected] / norm_factors
                y = test_data[:, -1][:, None]  # RUL

                test_x, test_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
                y_true_tmp, y_pred_tmp = [], []

                with torch.no_grad():
                    cycle = test_data[seq_length:, 0]

                    y_ = model(test_x) * rul_factor

                    y_true_tmp.append(test_y)
                    y_pred_tmp.append(y_)
                    y_true_tmp = torch.cat(y_true_tmp, axis=0)
                    y_pred_tmp = torch.cat(y_pred_tmp, axis=0)

                    rmse_loss_tmp = np.sqrt(mean_squared_error(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy()))
                    r2_tmp = r2_score(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy())
                    if not np.isnan(rmse_loss_tmp) and not np.isnan(r2_tmp):
                        rmse_scores_test_tmp.append(rmse_loss_tmp)
                        r2_scores_test_tmp.append(r2_tmp)

            if rmse_scores_test_tmp and r2_scores_test_tmp:
                average_rmse_test_tmp = np.mean(rmse_scores_test_tmp)
                average_r2_test_tmp = np.mean(r2_scores_test_tmp)
            else:
                average_rmse_test_tmp = np.nan
                average_r2_test_tmp = np.nan
            print(f"Epoch: {epoch}, Average RMSE: {average_rmse_test_tmp}, Average R2: {average_r2_test_tmp}")

            loss_test_list.append(average_rmse_test_tmp)

    # plt.plot(loss_train_list, c="blue", label="Train loss")
    # plt.plot(loss_test_list, c="red", label="Test Loss")
    # plt.title(f"Train and Test Loss")
    # plt.xlabel("10 Epochs")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig(os.path.join(iteration_path, f"Train and Test Loss.png"), dpi=200)
    # plt.close()

    model_name = f"gru_{iteration}"
    model_save_path = model_path + f"{iteration}/"

    try:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
    except OSError:
        print('Error: Creating directory. ' + model_save_path)

    torch.save(model, model_save_path + model_name)

    # # Evaluating the model
    # model = torch.load(model_path + model_name)

    for file_name in test_list:
        test_data = torch.load(data_path + file_name)

        x = test_data[:, num_selected] / norm_factors
        y = test_data[:, -1][:, None]  # RUL

        test_x, test_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            cycle = test_data[seq_length:, 0]

            y_ = model(test_x) * rul_factor

            y_true.append(test_y)
            y_pred.append(y_)
            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            rmse_loss = np.sqrt(mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()))
            r2 = r2_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

            rmse_scores.append(rmse_loss)
            r2_scores.append(r2)
            rmse_dict[file_name] = rmse_loss  # Store RMSE with its corresponding data_name
            r2_dict[file_name] = r2

        # plt.plot(y_true.detach().cpu().numpy(), y_true.detach().cpu().numpy(), c="red", label="True RUL")
        # plt.plot(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), c="blue", label="Predicted RUL")
        # plt.title(f"RUL Prediction of the Battery Cell #{file_name[-4:]}")
        # plt.xlabel("True RUL")
        # plt.ylabel("Predicted RUL")
        # plt.legend()
        # plt.savefig(os.path.join(iteration_path, f"Cell #{file_name[-4:]}.png"), dpi=200)
        # plt.close()

        with open(results_filename, "a") as file:
            file.write(f"Iteration {iteration} // {file_name}\n")
            file.write(f"RMSE: {rmse_loss}\n")
            file.write(f"R2: {r2}\n\n")

    average_rmse = np.mean(rmse_scores)
    print(average_rmse)
    std_dev_rmse = np.std(rmse_scores)
    average_r2 = np.mean(r2_scores)
    std_dev_r2 = np.std(r2_scores)
    max_rmse_data_name = max(rmse_dict, key=rmse_dict.get)
    min_r2_data_name = min(r2_dict, key=r2_dict.get)


    with open(results_filename, "a") as file:
        file.write(f"Iteration {iteration}, Selected Features: {num_selected}, Split_Seed: {seed},"
                   f" Epochs: {epochs}, Sequence Length: {seq_length}\n")
        file.write(f"Average RMSE: {average_rmse}, Std Dev: {std_dev_rmse}\n")
        file.write(f"Average R2: {average_r2}, Std Dev: {std_dev_r2}\n")
        file.write(f"Data Name with Highest RMSE: {max_rmse_data_name}, RMSE: {rmse_dict[max_rmse_data_name]}\n")
        file.write(f"Data Name with Smallest R2: {min_r2_data_name}, R2: {r2_dict[min_r2_data_name]}\n\n")
        file.write("=============================================================================\n\n")

    all_results_gru.append({
        # "Feature Index": num_selected,
        "Iteration": iteration,
        "Average RMSE": average_rmse,
        "RMSE Std Dev": std_dev_rmse,
        "Average R2": average_r2,
        "R2 Std Dev": std_dev_r2,
        "Max RMSE Data Name": max_rmse_data_name,
        "Max RMSE Value": rmse_dict[max_rmse_data_name],
        "Min R2 Data Name": min_r2_data_name,
        "Min R2 Value": r2_dict[min_r2_data_name]
    })
    print(f"[GRU] Experiment {iteration} completed and results saved.")
    print("")


def run_experiment_lstm(iteration, num_selected, seed, seq_length):
    ####################
    epochs = 1000
    train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=seed)

    loss_train_list = []
    loss_test_list = []
    ####################
    print(f"[LSTM] Experiment {iteration} started.")

    results_filename = os.path.join(result_path, f"results_lstm.txt")

    # iteration_path = os.path.join(result_path, f"LSTM_iter_{iteration}") # Create a directory for the current iteration
    # os.makedirs(iteration_path, exist_ok=True)

    rmse_scores = []
    r2_scores = []
    rmse_dict = {}  # To store RMSE for each data_name
    r2_dict = {}    # To store R2 for each data_name

    # train
    model = RNN_LSTM(input_dim=len(num_selected), hidden_dim=10, output_dim=1, layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    for epoch in range(epochs):
        random.shuffle(train_list)
        epoch_loss = 0.0
        num_data = len(train_list)
        for data_name in train_list:
            data = torch.load(data_path + data_name)

            x = data[:, num_selected] / norm_factors
            y = data[:, -1][:, None] / rul_factor  # RUL

            train_x, train_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)

            loss = model.loss(train_x, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu().numpy()

        scheduler.step()

        if epoch % 10 == 9:  # Evaluating the test set for every 10 epochs
            rmse_scores_train_tmp = []
            rmse_scores_test_tmp = []
            r2_scores_test_tmp = []

            ############## Train Files ##############
            for data_name in train_list:
                data = torch.load(data_path + data_name)

                x = data[:, num_selected] / norm_factors
                y = data[:, -1][:, None]  # RUL

                train_x, train_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
                y_true_tmp, y_pred_tmp = [], []

                with torch.no_grad():
                    cycle = data[seq_length:, 0]

                    y_ = model(train_x) * rul_factor

                    y_true_tmp.append(train_y)
                    y_pred_tmp.append(y_)
                    y_true_tmp = torch.cat(y_true_tmp, axis=0)
                    y_pred_tmp = torch.cat(y_pred_tmp, axis=0)

                    rmse_loss_tmp = np.sqrt(
                        mean_squared_error(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy()))
                    rmse_scores_train_tmp.append(rmse_loss_tmp)

            average_rmse_train_tmp = np.mean(rmse_scores_train_tmp)

            loss_train_list.append(average_rmse_train_tmp)

            ############## Test Files ##############
            for file_name in test_list:
                test_data = torch.load(data_path + file_name)

                x = test_data[:, num_selected] / norm_factors
                y = test_data[:, -1][:, None]  # RUL

                test_x, test_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
                y_true_tmp, y_pred_tmp = [], []

                with torch.no_grad():
                    cycle = test_data[seq_length:, 0]

                    y_ = model(test_x) * rul_factor

                    y_true_tmp.append(test_y)
                    y_pred_tmp.append(y_)
                    y_true_tmp = torch.cat(y_true_tmp, axis=0)
                    y_pred_tmp = torch.cat(y_pred_tmp, axis=0)

                    rmse_loss_tmp = np.sqrt(
                        mean_squared_error(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy()))
                    r2_tmp = r2_score(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy())
                    if not np.isnan(rmse_loss_tmp) and not np.isnan(r2_tmp):
                        rmse_scores_test_tmp.append(rmse_loss_tmp)
                        r2_scores_test_tmp.append(r2_tmp)

            if rmse_scores_test_tmp and r2_scores_test_tmp:
                average_rmse_test_tmp = np.mean(rmse_scores_test_tmp)
                average_r2_test_tmp = np.mean(r2_scores_test_tmp)
            else:
                average_rmse_test_tmp = np.nan
                average_r2_test_tmp = np.nan
            print(f"Epoch: {epoch}, Average RMSE: {average_rmse_test_tmp}, Average R2: {average_r2_test_tmp}")

            loss_test_list.append(average_rmse_test_tmp)

        # plt.plot(loss_train_list, c="blue", label="Train loss")
        # plt.plot(loss_test_list, c="red", label="Test Loss")
        # plt.title(f"Train and Test Loss")
        # plt.xlabel("10 Epochs")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.savefig(os.path.join(iteration_path, f"Train and Test Loss.png"), dpi=200)
        # plt.close()

    model_name = f"lstm_{iteration}"
    model_save_path = model_path + f"{iteration}/"

    try:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
    except OSError:
        print('Error: Creating directory. ' + model_save_path)

    torch.save(model, model_save_path + model_name)

    # # Evaluating the model
    # model = torch.load(model_path + model_name)

    for file_name in test_list:
        test_data = torch.load(data_path + file_name)

        x = test_data[:, num_selected] / norm_factors
        y = test_data[:, -1][:, None]  # RUL

        test_x, test_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():

            y_ = model(test_x) * rul_factor

            y_true.append(test_y)
            y_pred.append(y_)
            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            rmse_loss = np.sqrt(mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()))
            r2 = r2_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

            rmse_scores.append(rmse_loss)
            r2_scores.append(r2)
            rmse_dict[file_name] = rmse_loss  # Store RMSE with its corresponding data_name
            r2_dict[file_name] = r2

        # plt.plot(y_true.detach().cpu().numpy(), y_true.detach().cpu().numpy(), c="red", label="True RUL")
        # plt.plot(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), c="blue", label="Predicted RUL")
        # plt.title(f"RUL Prediction of the Battery Cell #{file_name[-4:]}")
        # plt.xlabel("True RUL")
        # plt.ylabel("Predicted RUL")
        # plt.legend()
        # plt.savefig(os.path.join(iteration_path, f"Cell #{file_name[-4:]}.png"), dpi=200)
        # plt.close()

        with open(results_filename, "a") as file:
            file.write(f"Iteration {iteration} // {file_name}\n")
            file.write(f"RMSE: {rmse_loss}\n")
            file.write(f"R2: {r2}\n\n")

    average_rmse = np.mean(rmse_scores)
    print(average_rmse)
    std_dev_rmse = np.std(rmse_scores)
    average_r2 = np.mean(r2_scores)
    std_dev_r2 = np.std(r2_scores)
    max_rmse_data_name = max(rmse_dict, key=rmse_dict.get)
    min_r2_data_name = min(r2_dict, key=r2_dict.get)


    with open(results_filename, "a") as file:
        file.write(f"Iteration {iteration}, Selected Features: {num_selected}, Split_Seed: {seed},"
                   f" Epochs: {epochs}, Sequence Length: {seq_length}\n")
        file.write(f"Average RMSE: {average_rmse}, Std Dev: {std_dev_rmse}\n")
        file.write(f"Average R2: {average_r2}, Std Dev: {std_dev_r2}\n")
        file.write(f"Data Name with Highest RMSE: {max_rmse_data_name}, RMSE: {rmse_dict[max_rmse_data_name]}\n")
        file.write(f"Data Name with Smallest R2: {min_r2_data_name}, R2: {r2_dict[min_r2_data_name]}\n\n")
        file.write("=============================================================================\n\n")

    all_results_lstm.append({
        # "Feature Index": num_selected,
        "Iteration": iteration,
        "Average RMSE": average_rmse,
        "RMSE Std Dev": std_dev_rmse,
        "Average R2": average_r2,
        "R2 Std Dev": std_dev_r2,
        "Max RMSE Data Name": max_rmse_data_name,
        "Max RMSE Value": rmse_dict[max_rmse_data_name],
        "Min R2 Data Name": min_r2_data_name,
        "Min R2 Value": r2_dict[min_r2_data_name]
    })
    print(f"[LSTM] Experiment {iteration} completed and results saved.")
    print("")


def run_experiment_transformer(iteration, num_selected, seed, seq_length):
    ####################
    epochs = 1000
    train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=seed)

    loss_train_list = []
    loss_test_list = []
    ####################
    print(f"[Transformer] Experiment {iteration} started.")

    results_filename = os.path.join(result_path, f"results_transformer.txt")

    # iteration_path = os.path.join(result_path, f"Transformer_iter_{iteration}") # Create a directory for the current iteration
    # os.makedirs(iteration_path, exist_ok=True)

    rmse_scores = []
    r2_scores = []
    rmse_dict = {}  # To store RMSE for each data_name
    r2_dict = {}    # To store R2 for each data_name

    # train
    model = TransformerModel(input_dim=len(num_selected), hidden_dim=32, output_dim=1, num_layers=3, nhead=8, dropout=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    for epoch in range(epochs):
        random.shuffle(train_list)
        epoch_loss = 0.0
        num_data = len(train_list)
        for data_name in train_list:
            data = torch.load(data_path + data_name)

            x = data[:, num_selected] / norm_factors
            y = data[:, -1][:, None] / rul_factor  # RUL

            train_x, train_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)

            loss = model.loss(train_x, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu().numpy()

        scheduler.step()

        if epoch % 10 == 9:  # Evaluating the test set for every 10 epochs
            rmse_scores_train_tmp = []
            rmse_scores_test_tmp = []
            r2_scores_test_tmp = []

            ############## Train Files ##############
            for data_name in train_list:
                data = torch.load(data_path + data_name)

                x = data[:, num_selected] / norm_factors
                y = data[:, -1][:, None]  # RUL

                train_x, train_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
                y_true_tmp, y_pred_tmp = [], []

                with torch.no_grad():
                    cycle = data[seq_length:, 0]

                    y_ = model(train_x) * rul_factor

                    y_true_tmp.append(train_y)
                    y_pred_tmp.append(y_)
                    y_true_tmp = torch.cat(y_true_tmp, axis=0)
                    y_pred_tmp = torch.cat(y_pred_tmp, axis=0)

                    rmse_loss_tmp = np.sqrt(
                        mean_squared_error(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy()))
                    rmse_scores_train_tmp.append(rmse_loss_tmp)

            average_rmse_train_tmp = np.mean(rmse_scores_train_tmp)

            loss_train_list.append(average_rmse_train_tmp)

            ############## Test Files ##############
            for file_name in test_list:
                test_data = torch.load(data_path + file_name)

                x = test_data[:, num_selected] / norm_factors
                y = test_data[:, -1][:, None]  # RUL

                test_x, test_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
                y_true_tmp, y_pred_tmp = [], []

                with torch.no_grad():
                    cycle = test_data[seq_length:, 0]

                    y_ = model(test_x) * rul_factor

                    y_true_tmp.append(test_y)
                    y_pred_tmp.append(y_)
                    y_true_tmp = torch.cat(y_true_tmp, axis=0)
                    y_pred_tmp = torch.cat(y_pred_tmp, axis=0)

                    rmse_loss_tmp = np.sqrt(
                        mean_squared_error(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy()))
                    r2_tmp = r2_score(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy())
                    if not np.isnan(rmse_loss_tmp) and not np.isnan(r2_tmp):
                        rmse_scores_test_tmp.append(rmse_loss_tmp)
                        r2_scores_test_tmp.append(r2_tmp)

            if rmse_scores_test_tmp and r2_scores_test_tmp:
                average_rmse_test_tmp = np.mean(rmse_scores_test_tmp)
                average_r2_test_tmp = np.mean(r2_scores_test_tmp)
            else:
                average_rmse_test_tmp = np.nan
                average_r2_test_tmp = np.nan
            print(f"Epoch: {epoch}, Average RMSE: {average_rmse_test_tmp}, Average R2: {average_r2_test_tmp}")

            loss_test_list.append(average_rmse_test_tmp)

        # plt.plot(loss_train_list, c="blue", label="Train loss")
        # plt.plot(loss_test_list, c="red", label="Test Loss")
        # plt.title(f"Train and Test Loss")
        # plt.xlabel("10 Epochs")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.savefig(os.path.join(iteration_path, f"Train and Test Loss.png"), dpi=200)
        # plt.close()

    model_name = f"Transformer_{iteration}"
    model_save_path = model_path + f"{iteration}/"

    try:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
    except OSError:
        print('Error: Creating directory. ' + model_save_path)

    torch.save(model, model_save_path + model_name)

    # # Evaluating the model
    # model = torch.load(model_path + model_name)

    for file_name in test_list:
        test_data = torch.load(data_path + file_name)

        x = test_data[:, num_selected] / norm_factors
        y = test_data[:, -1][:, None]  # RUL

        test_x, test_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            cycle = test_data[seq_length:, 0]

            y_ = model(test_x) * rul_factor

            y_true.append(test_y)
            y_pred.append(y_)
            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            rmse_loss = np.sqrt(mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()))
            r2 = r2_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

            rmse_scores.append(rmse_loss)
            r2_scores.append(r2)
            rmse_dict[file_name] = rmse_loss  # Store RMSE with its corresponding data_name
            r2_dict[file_name] = r2

        # plt.plot(y_true.detach().cpu().numpy(), y_true.detach().cpu().numpy(), c="red", label="True RUL")
        # plt.plot(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), c="blue", label="Predicted RUL")
        # plt.title(f"RUL Prediction of the Battery Cell #{file_name[-4:]}")
        # plt.xlabel("True RUL")
        # plt.ylabel("Predicted RUL")
        # plt.legend()
        # plt.savefig(os.path.join(iteration_path, f"Cell #{file_name[-4:]}.png"), dpi=200)
        # plt.close()

        with open(results_filename, "a") as file:
            file.write(f"Iteration {iteration} // {file_name}\n")
            file.write(f"RMSE: {rmse_loss}\n")
            file.write(f"R2: {r2}\n\n")

    average_rmse = np.mean(rmse_scores)
    print(average_rmse)
    std_dev_rmse = np.std(rmse_scores)
    average_r2 = np.mean(r2_scores)
    std_dev_r2 = np.std(r2_scores)
    max_rmse_data_name = max(rmse_dict, key=rmse_dict.get)
    min_r2_data_name = min(r2_dict, key=r2_dict.get)


    with open(results_filename, "a") as file:
        file.write(f"Iteration {iteration}, Selected Features: {num_selected}, Split_Seed: {seed},"
                   f" Epochs: {epochs}, Sequence Length: {seq_length}\n")
        file.write(f"Average RMSE: {average_rmse}, Std Dev: {std_dev_rmse}\n")
        file.write(f"Average R2: {average_r2}, Std Dev: {std_dev_r2}\n")
        file.write(f"Data Name with Highest RMSE: {max_rmse_data_name}, RMSE: {rmse_dict[max_rmse_data_name]}\n")
        file.write(f"Data Name with Smallest R2: {min_r2_data_name}, R2: {r2_dict[min_r2_data_name]}\n\n")
        file.write("=============================================================================\n\n")

    all_results_transformer.append({
        # "Feature Index": num_selected,
        "Iteration": iteration,
        "Average RMSE": average_rmse,
        "RMSE Std Dev": std_dev_rmse,
        "Average R2": average_r2,
        "R2 Std Dev": std_dev_r2,
        "Max RMSE Data Name": max_rmse_data_name,
        "Max RMSE Value": rmse_dict[max_rmse_data_name],
        "Min R2 Data Name": min_r2_data_name,
        "Min R2 Value": r2_dict[min_r2_data_name]
    })
    print(f"[Transformer] Experiment {iteration} completed and results saved.")
    print("")

for i in range(1):
    seed = 17 * i
    num_selected = [2, 1, 146, 57, 141, 68, 25, 99, 95, 87, 78, 108, 152]

    run_experiment_mlp(i, num_selected, seed, seq_length)
    run_experiment_gru(i, num_selected, seed, seq_length)
    run_experiment_lstm(i, num_selected, seed, seq_length)
    run_experiment_transformer(i, num_selected, seed, seq_length)


with open(csv_filename_mlp, 'w', newline='') as csvfile:
    fieldnames = list(all_results_mlp[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in all_results_mlp:
        writer.writerow(result)

with open(csv_filename_gru, 'w', newline='') as csvfile:
    fieldnames = list(all_results_gru[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in all_results_gru:
        writer.writerow(result)


with open(csv_filename_lstm, 'w', newline='') as csvfile:
    fieldnames = list(all_results_lstm[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in all_results_lstm:
        writer.writerow(result)


with open(csv_filename_transformer, 'w', newline='') as csvfile:
    fieldnames = list(all_results_transformer[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in all_results_transformer:
        writer.writerow(result)

print("All results saved to CSV.")