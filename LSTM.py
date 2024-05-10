import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PreProcessing import Preprocessing
import torch.nn as nn
import torch.optim as optim
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def nnModel():
    df = pd.read_csv("Amazon.csv")

    preprocessor = Preprocessing(df)

    df = preprocessor.outlierRemoval()

    df.loc[:, 'Date'] = pd.to_datetime(df['Date']).copy()
    df = df[['Date', 'Close']]

    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop(columns=['Date'], inplace=True)

    scaler = StandardScaler()
    df[['Year', 'Month', 'Day']] = scaler.fit_transform(df[['Year', 'Month', 'Day']])

    x = df.drop(columns=['Close'])
    y = df['Close']

    X_tensor = torch.tensor(x.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    train_size = int(0.8 * len(x))
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

    class SimpleRegressionNN(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(SimpleRegressionNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.fc3 = nn.Linear(hidden_size2, output_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    input_size = 3
    hidden_size1 = 64
    hidden_size2 = 64
    output_size = 1
    model = SimpleRegressionNN(input_size=3, hidden_size1=128, hidden_size2=64, output_size=1)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30000
    for epoch in range(num_epochs):

        outputs = model(X_train)
        loss = criterion(outputs, y_train.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate model
    with torch.no_grad():
        predicted = model(X_test)
        test_loss = criterion(predicted, y_test.unsqueeze(1))
        print(f'Test Loss: {test_loss.item():.4f}')

        predicted_np = predicted.squeeze().numpy()
        y_test_np = y_test.squeeze().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(predicted_np, label='Predicted')
        plt.plot(y_test_np, label='True')
        plt.xlabel('Index')
        plt.ylabel('Close Value')
        plt.title('Comparison between Predicted and True Values')
        plt.legend()
        plt.show()