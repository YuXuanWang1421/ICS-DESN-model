import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ------------------- ETTh1 数据生成/读取 -------------------
def generate_etth1_data(file_path, target_col='OT', sequence_length=12):
    df = pd.read_csv(file_path)

    # 保留数值列
    df_numeric = df.select_dtypes(include=[np.number])

    if target_col not in df_numeric.columns:
        raise ValueError(f"CSV中未找到目标列 {target_col}")

    feature_cols = [c for c in df_numeric.columns if c != target_col]

    features = df_numeric[feature_cols].values.astype(float)
    target = df_numeric[target_col].values.astype(float).reshape(-1, 1)

    # 归一化
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler_X.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target)

    # 构建滑动窗口序列
    X, y = [], []
    for i in range(len(features_scaled) - sequence_length):
        X.append(features_scaled[i:i+sequence_length])
        y.append(target_scaled[i+sequence_length])

    X = np.array(X)
    y = np.array(y)

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, shuffle=False
    )

    # 转换为 torch tensor
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


# ------------------- LSTM 模型 -------------------
class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1):
        super(StackedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


# ------------------- 模型训练 -------------------
def train_model(model, X_train, y_train, epochs=100, lr=0.001, device='cpu'):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_train, y_train = X_train.to(device), y_train.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    return model


# ------------------- 模型评估 -------------------
def evaluate_model(model, X, y, scaler_y, device, title="结果", save_csv=None):
    model.eval()
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        preds = model(X).cpu().numpy()
        trues = y.cpu().numpy()
        errors = preds - trues

        mse = mean_squared_error(trues, preds)
        mae = mean_absolute_error(trues, preds)
        r2 = r2_score(trues, preds)
        mse_var = np.var(errors**2)
        mae_var = np.var(np.abs(errors))

        print(f"{title}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
        print(f"方差: MSE_var={mse_var:.12f}, MAE_var={mae_var:.12f}")

        if save_csv is not None:
            df = pd.DataFrame({
                "True": trues.flatten(),
                "Pred": preds.flatten(),
                "Error": errors.flatten()
            })
            df.to_csv(save_csv, index=False, encoding="utf-8-sig")
            print(f"已保存 {save_csv}")

        plt.figure(figsize=(12,5))
        plt.plot(trues, label="True")
        plt.plot(preds, label="Pred", alpha=0.7)
        plt.title(title)
        plt.legend()
        plt.show()

    return preds, trues, errors


# ------------------- 主函数 -------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    file_path = "./ETTh1.csv"
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = generate_etth1_data(file_path, target_col='OT', sequence_length=12)

    input_size = X_train.shape[2]
    model = StackedLSTM(input_size=input_size, hidden_size=256, num_layers=3, output_size=1).to(device)

    print("开始训练...")
    model = train_model(model, X_train, y_train, epochs=120, lr=0.001, device=device)

    print("训练集评估...")
    evaluate_model(model, X_train, y_train, scaler_y, device, title="训练集", save_csv="etth1_train_results.csv")

    print("测试集评估...")
    evaluate_model(model, X_test, y_test, scaler_y, device, title="测试集", save_csv="etth1_test_results.csv")


if __name__ == "__main__":
    main()
