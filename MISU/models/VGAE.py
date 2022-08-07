import torch
from torch_geometric.nn.models.autoencoder import VGAE, MAX_LOGSTD
# my files
from MISU.utils import get_junction_tree

EPS = 1e-15


class VGAENoNeg(VGAE):

    def encode(self, batch, device, fgvae=False):
        """"""
        self.__mu__, self.__logstd__ = self.encoder(batch)
        if fgvae:
            self.__mu__, _, _ = get_junction_tree(self.__mu__, batch, device)
            self.__logstd__, _, _ = get_junction_tree(self.__logstd__, batch, device)
            # self.__mu__, self.__logstd__ = fg_embs, fg_embs
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
                entropy loss for positive edges :obj:`pos_edge_index` and negative
                sampled edges.

                Args:
                    z (Tensor): The latent space :math:`\mathbf{Z}`.
                    pos_edge_index (LongTensor): The positive edges to train against.
                    neg_edge_index (LongTensor, optional): The negative edges to train
                        against. If not given, uses negative sampling to calculate
                        negative edges. (default: :obj:`None`)
                """

        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        loss = pos_loss
        if neg_edge_index is not None:
            neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
            loss += neg_loss
        return loss