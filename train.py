import multiprocessing
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from intrusion_detection_datasets import UNSW_NB15Dataset
from models import BYOL, resnet, densenet, botnet, squeezenet, mobilenet, seresnet
from utils import  mse_loss


# hyperparameter
BATCH_SIZE = 512
EPOCHS = 6
LR = 3e-4
IMAGE_SIZE = 14
NUM_WORKERS = multiprocessing.cpu_count()


# pytorch lightning module
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, training_loss, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)
        self.training_loss = training_loss

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        self.training_loss.append(loss)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        self.learner.update_moving_average()


# main
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_csv_path = './datasets/UNSW_NB15_training-set.csv'
    test_csv_path = './datasets/UNSW_NB15_testing-set.csv'
    ds = UNSW_NB15Dataset(train_csv_path, test_csv_path, IMAGE_SIZE)
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    net = botnet([32, 64, 128, 256])
    mse_list = []
   
    model = SelfSupervisedLearner(
        net,
        mse_list,
        image_size=IMAGE_SIZE,
        loss_fn=mse_loss,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=128,
        moving_average_decay=0.99
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=EPOCHS,
        accumulate_grad_batches=1,
        sync_batchnorm=True
    )

    trainer.fit(model, train_loader)
    torch.save(net.state_dict(), 'Representation.pt')
    plt.figure(dpi=300)
    plt.xlabel('Training steps')
    plt.ylabel('Training loss')
    plt.plot(mse_list, color='blue', linestyle='-', label='MSE loss')
    plt.legend()
    plt.show()

    