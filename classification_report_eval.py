
import os
import torch
import numpy as np
from cnn_model import FallDetectionCNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

def load_data(data_dir):
    X, y = [], []
    for label_name in ['fall', 'nonfall']:
        label = 1 if label_name == 'fall' else 0
        folder = os.path.join(data_dir, label_name)
        for fname in os.listdir(folder):
            if fname.endswith('.npy'):
                data = np.load(os.path.join(folder, fname))
                if data.shape == (30, 66):
                    X.append(data)
                    y.append(label)
    return X, y

def predict(model, X):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for sample in X:
            input_tensor = torch.tensor([sample], dtype=torch.float32)
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            y_pred.append(pred)
    return y_pred

data_dir = "pose_data"
X, y_true = load_data(data_dir)

model = FallDetectionCNN()
model.load_state_dict(torch.load("fall_cnn_model.pth", map_location=torch.device("cpu")))

y_pred = predict(model, X)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["NON-FALL", "FALL"]))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NON-FALL", "FALL"])
disp.plot(cmap="Blues")
plt.title("Fall Detection Confusion Matrix")
plt.show()
