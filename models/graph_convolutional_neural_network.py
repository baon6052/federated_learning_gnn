import lightning as L
import torch
from torch import nn, optim
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(L.LightningModule):
    def __init__(
        self, num_features: int, num_classes: int, visualise: bool = False
    ):
        super().__init__()

        self.visualise = visualise

        torch.manual_seed(1234)
        self.conv1 = GCNConv(num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        out = self.classifier(h)
        return out, h

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        out, h = self.forward(batch.x, batch.edge_index)
        out = out.cpu()
        batch = batch.cpu()
        loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])

        return loss

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, "test")

    def _evaluate(self, batch, stage: str):
        out, h = self.forward(batch.x, batch.edge_index)
        out = out.cpu()
        batch = batch.cpu()

        node_mask = batch.val_mask
        label_mask = batch.val_mask

        if stage == "test":
            node_mask = batch.test_mask
            label_mask = batch.test_mask

        pred = out.argmax(1)

        # loss = self.criterion(out[node_mask], batch.y[label_mask])
        acc = (pred[node_mask] == batch.y[label_mask]).float().mean()
        self.log(f"{stage}_loss", acc, prog_bar=False)
