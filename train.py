import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import utils
import transformer as tr
import os
import datetime
import json
from pathlib import Path
import joblib
import pandas as pd
import argparse
import warnings
from torch.utils.tensorboard import SummaryWriter

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'

# Model parameters
package_dir = Path(__file__).parent.absolute()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a JSON file and an csv files.")
    parser.add_argument('--metadata', type=str, required=True, help='Path to the JSON file')
    
    args = parser.parse_args()

    with open(args.metadata) as f:
        metaData = json.load(f)

    ################################>>>>>>>>>>>>>>>>CONFIG<<<<<<<<<<<<<######################################################

    basePath = package_dir
   
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    path_to_save = os.path.join(package_dir, TIMESTAMP)
    os.mkdir(path_to_save)

    req_input_signals = metaData['INPUT_SIGNALS'] #['signal1', 'signal2', 'signal3']   ##step is mandatory data point for this trained drive cycle
    req_output_signals = metaData['OUTPUT_SIGNALS'] #['Output1','Output2','Output3']

    reqSignals = req_input_signals + req_output_signals

    EPOCHS = metaData['EPOCHS']
    BATCH_SIZE = metaData['BATCH_SIZE']
    VAL_SPLIT = metaData['VAL_SPLIT']

    nhead = metaData['N_HEAD']
    d_model = metaData['D_MODEL']
    num_epochs = metaData['EPOCHS']
    num_encoder_layers = metaData['ENCODER_LAYERS']
    num_decoder_layers = metaData['DECODER_LAYERS']
    dim_feedforward = metaData['FEEDFRWD_DIM']
    learning_rate = metaData['LR']
    early_stopping_patience = metaData['PATIENCE']

    log_dir = "./runs/transformer"
    checkpoint_dir = "./checkpoints"

    # Ensure directories exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(TIMESTAMP, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    TRAINED_MODEL_NAME = "Transformer_{}_{}_{}_{}_{}.pth".format(metaData['MODEL_NAME'],
                                                        metaData['EPOCHS'],
                                                        metaData['ENCODER_LAYERS'],
                                                        metaData['DECODER_LAYERS'],
                                                        TIMESTAMP)

    TRAINED_MODEL_NAME = f"Transformer_{metaData['MODEL_NAME']}_{metaData['EPOCHS']}_{metaData['ENCODER_LAYERS']}_{metaData['DECODER_LAYERS']}_{TIMESTAMP}"

    trainDataFrame = utils.read_csv_file(metaData['TRAIN_DATA_PATH'],reqSignals)
    testDataFrame = utils.read_csv_file(metaData['TEST_DATA_PATH'], reqSignals)

    transformedTrainData, input_scaler_train, output_scaler_train = utils.scale_data(trainDataFrame,req_input_signals, req_output_signals)
    transformedTestData, input_scaler_test, output_scaler_test = utils.scale_data(testDataFrame,req_input_signals, req_output_signals)

    INPUT_SCALER_NAME = f'inputDataScalerTrain_{TIMESTAMP}.save'
    OUTPUT_SCALER_NAME = f'outputDataScalerTrain_{TIMESTAMP}.save'

    joblib.dump(input_scaler_train, os.path.join(path_to_save,INPUT_SCALER_NAME))
    joblib.dump(output_scaler_train, os.path.join(path_to_save,OUTPUT_SCALER_NAME))

    inputVector = utils.sliding_window(transformedTrainData[req_input_signals].to_numpy(), window_size=512, step=512)
    outputVector = utils.sliding_window(transformedTrainData[req_output_signals].to_numpy(), window_size=512, step=512)

    print("shape of Input vector is {}".format(inputVector.shape))
    print("shape of Output vector is {}".format(outputVector.shape))

    input_dim = inputVector.shape[2]
    output_dim = outputVector.shape[2]

    print("shape of Input dim  is {}".format(input_dim))
    print("shape of Output dim  is {}".format(output_dim))

    dataset = tr.TimeSeriesDataset(inputVector, outputVector)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    model = tr.TimeSeriesTransformer(input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
    writer = SummaryWriter(log_dir)

    best_loss = float('inf')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            input_sequence = batch['input_sequence'].transpose(0,1).to(device)  # Adding batch dimension
            output_sequence = batch['output_sequence'].transpose(0,1).to(device)  # Adding batch dimension
            optimizer.zero_grad()
            output = model(input_sequence, output_sequence)
            loss = criterion(output, output_sequence)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        model.eval()
        all_outputs, all_targets = [], []
        with torch.no_grad():
            for batch in dataloader:
                input_sequence = batch['input_sequence'].transpose(0, 1).to(device)
                output_sequence = batch['output_sequence'].transpose(0, 1).to(device)
                output = model(input_sequence, output_sequence)
                all_outputs.append(output)
                all_targets.append(output_sequence)
            all_outputs = torch.cat(all_outputs, dim=1)  # Concatenate along batch dimension
            all_targets = torch.cat(all_targets, dim=1)  # Concatenate along batch dimension

        mse, mae, mape, accuracy = utils.compute_metrics(all_outputs, all_targets)
        
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        writer.add_scalar('Metrics/MAE', mae, epoch)
        writer.add_scalar('Metrics/MSE', mse, epoch)
        writer.add_scalar('Metrics/MAPE', mape, epoch)
        writer.add_scalar('Metrics/Accuracy', accuracy, epoch)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(path_to_save, "best_"+TRAINED_MODEL_NAME))
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch}')
            break

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

    writer.close()

    with open(os.path.join(path_to_save,f"Transformer_metadata_{TIMESTAMP}.json"), "w") as fp:
        json.dump(metaData , fp)

    torch.save(model.state_dict(), os.path.join(path_to_save,TRAINED_MODEL_NAME))

