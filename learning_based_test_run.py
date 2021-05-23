import numpy as np
import torch
from torch.utils.data import DataLoader
from learning_based_cnn_param_loader.dataloader import notMNIST
import os
from learning_based_cnn_param_loader.parameters import MODEL_NAME

# Path สำหรับโฟลเดอร์ dataset สำหรับ test
path = os.path.join(os.path.dirname(__file__), 'Dataset/test')
print(path)
test_dataset = notMNIST(path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
classifier = torch.load('model/{}.pt'.format(MODEL_NAME)).eval()
correct = 0

for _, data in enumerate(test_loader, 0):
	test_x, test_y = data
	pred = classifier.forward(test_x)
	y_hat = np.argmax(pred.data)
	if y_hat == test_y:
		print("classifier: {} / correct_ans: {} - correct".format(y_hat, test_y))
		correct += 1
	else:
		print("classifier: {} / correct_ans: {} - wrong".format(y_hat, test_y))

print("Accuracy={}".format(correct / len(test_dataset)))
