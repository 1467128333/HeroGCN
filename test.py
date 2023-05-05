from __future__ import print_function, division
import argparse
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from HeroGCN import AE, Summarizer, make_modularity_matrix, corruption

EPS = 1e-15


class HeroGCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(HeroGCN, self).__init__()

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = v

        self.summary = Summarizer()
        self.weight = Parameter(torch.Tensor(n_enc_3, n_enc_3))

    def modularity(self, p, adj):
        mod = make_modularity_matrix(adj)
        bin_adj_nodiag = adj * (torch.ones(adj.shape[0], adj.shape[0]).to(device) - torch.eye(adj.shape[0]).to(device))
        return (1. / bin_adj_nodiag.sum()) * (p.t() @ mod @ p).trace()

    def discriminate(self, z, summary, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss

    def forward(self, x, adj):
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        sigma = 0.5

        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        out_h = (1 - sigma) * h + sigma * tra3
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)
        predict = F.softmax(h, dim=1)

        neg_x = corruption(x)
        _, neg_tra1, neg_tra2, neg_tra3, neg_z = self.ae(neg_x)
        neg_h = self.gnn_1(x, adj)
        neg_h = self.gnn_2((1 - sigma) * neg_h + sigma * neg_tra1, adj)
        neg_h = self.gnn_3((1 - sigma) * neg_h + sigma * neg_tra2, adj)
        neg_h = (1 - sigma) * neg_h + sigma * neg_tra3
        summary = self.summary(out_h)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return h, out_h, neg_h, summary, x_bar, q, predict, z


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def test_herogcn(dataset):
    model = HeroGCN(500, 500, 2000, 2000, 500, 500,
                    n_input=args.n_input,
                    n_z=args.n_z,
                    n_clusters=args.n_clusters,
                    v=1.0).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    adj_all, adj = load_graph(args.name, args.k)
    adj = adj.cuda()
    adj_all = adj_all.cuda()

    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')

    model.load_state_dict(torch.load("my{}.pkl".format(args.name)))
    h, out_h, _, _, _, tmp_q, pred, z = model(data, adj)
    tmp_q = tmp_q.data
    p = target_distribution(tmp_q)
    res1 = tmp_q.cpu().numpy().argmax(1)
    res2 = pred.data.cpu().numpy().argmax(1)  # Z
    res3 = p.data.cpu().numpy().argmax(1)  # P
    acc_q = eva(y, res1, 'Q')
    acc_z = eva(y, res2, 'Z')
    acc_y = eva(y, res3, 'P')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='cite')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)  # 10 改为 300
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    args.pretrain_path = 'data/{}.pkl'.format(args.name)
    dataset = load_data(args.name)

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334

    print(args)
    test_herogcn(dataset)
