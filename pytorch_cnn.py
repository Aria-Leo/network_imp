from collections import defaultdict
from tqdm import tqdm

import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils


class CNNModule(torch.nn.Module):

    def __init__(self):
        super(CNNModule, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Linear(64, 10)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out


class PytorchCNN:

    def __init__(self):
        self.cnn_module = CNNModule()

    def fit(self, train_dataset, epochs=50, batch_size=50, validation_dataset=None):
        train_loader = data_utils.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn = self.cnn_module.to(device)
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(cnn.parameters(), lr=0.001)
        history = defaultdict(list)
        for epoch in range(epochs):
            total_cost, total_correct = 0, 0
            process_bar = tqdm(enumerate(train_loader))
            for step, im_data in process_bar:
                images, labels = im_data
                images = images.to(device)
                labels = labels.to(device)
                outputs = cnn(images)
                loss = loss_func(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, pred = outputs.max(1)
                correct = torch.sum(pred == labels).item()
                total_cost += loss.item()
                total_correct += correct
                process_bar.set_description(f'{epoch + 1} epochs, '
                                            f'batch {step}/{len(train_dataset) // batch_size}, '
                                            f'average loss is {loss.item()}, '
                                            f'current batch correct: {correct / batch_size:.2%}')
            average_cost = total_cost / (len(train_dataset) // batch_size)
            accuracy = total_correct / len(train_dataset)
            print(f'epoch {epoch} average cost on training data: {average_cost}')
            print(f'epoch {epoch} accuracy on training data: {accuracy:.2%}')
            history['loss'].append(average_cost)
            history['accuracy'].append(accuracy)

            if validation_dataset is not None:
                with torch.no_grad():
                    images, labels = validation_dataset.tensors
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = cnn(images)
                    loss = loss_func(outputs, labels).item()
                    _, pred = outputs.max(1)
                    correct = torch.sum(pred == labels).item()
                    accuracy = correct / len(validation_dataset)
                    print(f'epoch {epoch} average cost on validation data: {loss}')
                    print(f'epoch {epoch} accuracy on validation data: {accuracy:.2%}')
                    history['val_loss'].append(loss)
                    history['val_accuracy'].append(accuracy)
        return pd.DataFrame.from_dict(history)

    def predict(self, test_dataset):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            loss_func = torch.nn.CrossEntropyLoss()
            images, labels = test_dataset.tensors
            images = images.to(device)
            labels = labels.to(device)
            cnn = self.cnn_module.to(device)
            outputs = cnn(images)
            loss = loss_func(outputs, labels).item()
            _, pred = outputs.max(1)
            correct = torch.sum(pred == labels).item()
            print(f'average cost on test data: {loss}')
            print(f'accuracy on test data: {correct / len(test_dataset):.2%}')
        return pred.detach().cpu().numpy()
