from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd

import os
import ezkl
import json

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)


class TrainData(Dataset):
    """Train data class"""

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)


class TestData(Dataset):
    """Test data class"""

    def setX(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__ (self):
        return len(self.X_data)


class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 41.
        self.layer_1 = nn.Linear(41, 82)
        self.layer_2 = nn.Linear(82, 82)
        self.layer_out = nn.Linear(82, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.1)
        self.batchnorm1 = nn.BatchNorm1d(82)
        self.batchnorm2 = nn.BatchNorm1d(82)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


def pytorch_auc(y_pred, y_test):
    return roc_auc_score(y_test, y_pred.detach().cpu().numpy())


def test_integration():
    """Test integration with data/1_*_test/train.csv"""
    X_train = pd.read_csv(os.path.join(CURRENT_PATH, 'data', '1_X_train.csv'))
    y_train = pd.read_csv(os.path.join(CURRENT_PATH, 'data', '1_y_train.csv'))
    X_test = pd.read_csv(os.path.join(CURRENT_PATH, 'data', '1_X_test.csv'))
    y_test = pd.read_csv(os.path.join(CURRENT_PATH, 'data', '1_y_test.csv'))

    train_data = TrainData(torch.Tensor(X_train.to_numpy()), torch.Tensor(y_train.to_numpy()))
    test_data = TestData()
    test_data.setX(torch.Tensor(X_test.to_numpy()))

    data_loader = DataLoader(
        dataset = train_data,
        batch_sampler = StratifiedBatchSampler(torch.tensor(y_train.to_numpy().flatten()), batch_size = 8),
    )

    test_loader = DataLoader(dataset = test_data, batch_size = 2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize the model and optimizer
    model = BinaryClassification()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.005)

    model.train()
    for e in range(1, 5):
        epoch_loss = 0
        epoch_auc = 0
        for X_batch, y_batch in data_loader: #train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)
            AUROC = pytorch_auc(y_pred, y_batch.detach().cpu().numpy())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_auc += AUROC.item()


        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(data_loader):.5f} | AUC: {epoch_auc/len(data_loader):.3f}')

    y_pred_list = []
    y_pred_prob_list = []
    model.eval()
    with torch.no_grad():
        for X_test_batch in test_loader:
            X_test_batch = X_test_batch.to(device)
            y_test_pred = model(X_test_batch)
            y_test_pred_prob = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred_prob)
            y_pred_list.append(y_pred_tag.cpu().numpy())
            y_pred_prob_list.append(y_test_pred_prob.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_pred_prob_list = [a.squeeze().tolist() for a in y_pred_prob_list]


    ezkl.export(
        model.to('cpu'),
        input_shape = [41],
        onnx_filename = os.path.join(CURRENT_PATH, 'network_small.onnx'),
        input_filename = os.path.join(CURRENT_PATH, 'input_small.json'),
        settings_filename = os.path.join(CURRENT_PATH, 'settings.json')
    )

    # get log rows from settings.json
    with open(os.path.join(CURRENT_PATH, 'settings.json'), 'r') as f:
        data = json.load(f)

    ezkl.gen_srs(os.path.join(CURRENT_PATH, 'srs.params'), data['run_args']['logrows'])

    res = ezkl.setup(
        os.path.join(CURRENT_PATH, 'network_small.onnx'),
        os.path.join(CURRENT_PATH, 'model_vk_small.vk'),
        os.path.join(CURRENT_PATH, 'model_pk_small.pk'),
        os.path.join(CURRENT_PATH, 'srs.params'),
        os.path.join(CURRENT_PATH, 'settings.json')
    )
    assert res

    res = ezkl.prove(
        os.path.join(CURRENT_PATH, 'input_small.json'),
        os.path.join(CURRENT_PATH, 'network_small.onnx'),
        os.path.join(CURRENT_PATH, 'model_pk_small.pk'),
        os.path.join(CURRENT_PATH, 'zkml_proof_small.pf'),
        os.path.join(CURRENT_PATH, 'srs.params'),
        'poseidon',
        'single',
        os.path.join(CURRENT_PATH, 'settings.json'),
        False
    )

    assert res

    res = ezkl.mock(
        os.path.join(CURRENT_PATH, 'input_small.json'),
        os.path.join(CURRENT_PATH, 'network_small.onnx'),
        os.path.join(CURRENT_PATH, 'settings.json')
    )

    assert res

    res = ezkl.verify(
        os.path.join(CURRENT_PATH, 'zkml_proof_small.pf'),
        os.path.join(CURRENT_PATH, 'settings.json'),
        os.path.join(CURRENT_PATH, 'model_vk_small.vk'),
        os.path.join(CURRENT_PATH, 'srs.params'),
    )

    assert res