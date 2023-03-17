from ae_litmodel import LitAutoEncoder
from ae_dataloader import TimeSeriesDataset
import lightning.pytorch as pl
from torch import optim, nn, utils, Tensor


if __name__ == "__main__":
    # define any number of nn.Modules (or use your current ones)
    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    # init the autoencoder
    autoencoder = LitAutoEncoder(encoder, decoder)
    # setup data
    #dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    dataset = TimeSeriesDataset(data_dir="data/processed/")
    train_loader = utils.data.DataLoader(dataset)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)


    # load checkpoint
    checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
    autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

    # choose your trained nn.Module
    encoder = autoencoder.encoder
    encoder.eval()

    # embed 4 fake images!
    fake_image_batch = Tensor(4, 28 * 28)
    embeddings = encoder(fake_image_batch)
    print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)