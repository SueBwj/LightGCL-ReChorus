# -*- coding: UTF-8 -*-
# @Author : Wang Jie
# @Email  : wangj928@mail2.sysu.edu.cn

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

from models.BaseModel import GeneralModel


class LightGCLBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=2,
                            help='Number of LightGCL layers.')
        parser.add_argument('--q', type=int, default=5, help='rank')
        parser.add_argument('--temp', default=0.2, type=float,
                            help='temperature in cl loss')
        parser.add_argument('--lambda1', default=0.2,
                            type=float, help='weight of cl loss')
        parser.add_argument('--lambda2', default=1e-7,
                            type=float, help='l2 reg weight')
        parser.add_argument('--ssl_temp', default=0.2, type=float,
                          help='temperature in ssl loss')
        parser.add_argument('--ssl_reg', default=1e-5, type=float,
                          help='weight of ssl loss')
        parser.add_argument('--hyper_layers', type=int, default=1,
                          help='number of layers for hyper-graph')
        return parser

    @staticmethod
    def normalized_adj_single(adj):
        """ 
        计算归一化后的邻接矩阵
        """
        # D^-1/2 * A * D^-1/2
        rowsum = np.array(adj.sum(1)) + 1e-10

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    @staticmethod
    def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1.0
        R = R.tocoo()

        rowD = np.array(R.sum(1)).flatten() + 1e-10
        colD = np.array(R.sum(0)).flatten() + 1e-10

        R_coo = R.tocoo()
        data = R_coo.data
        row = R_coo.row
        col = R_coo.col

        for i in range(len(data)):
            data[i] = data[i] / pow(rowD[row[i]] * colD[col[i]], 0.5)

        norm_adj_mat = sp.csr_matrix(
            (data, (row, col)), shape=(user_count, item_count))

        return norm_adj_mat

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.norm_adj = self.build_adjmat(
            corpus.n_users, corpus.n_items, corpus.train_clicked_set)
        self.q = args.q
        self.temp = args.temp
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2        # 初始化一些参数
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = args.ssl_reg
        self.hyper_layers = args.hyper_layers
        self._base_define_params()
        self.apply(self.init_weights)

    def _base_define_params(self):
        self.encoder = LGCLEncoder(self.user_num, self.item_num, self.emb_size,
                                   self.norm_adj, self.n_layers, self.q, self.temp, self.lambda1, self.lambda2, self.hyper_layers)

    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']
        out_dict = self.encoder(user, items)  # 获取返回的字典
        user_embed = out_dict['user_embeddings']
        item_embed = out_dict['item_embeddings']
        g_user_embed = out_dict['G_user_embeddings']
        g_item_embed = out_dict['G_item_embeddings']
        context_embedding = out_dict['context_embedding']
        # user embed -> [batch_size, emb_size]
        # item embed -> [batch_size, num_items ,emb_size]
        # 在用户嵌入的中间添加一个维度，形状从[batch_size, embedding_dim]变为[batch_size, 1, embedding_dim]。
        prediction = (user_embed[:, None, :] * item_embed).sum(dim=-1)
        # 调整user_embed的维度使其为[batch_size, num_items ,emb_size]
        u_v = user_embed.repeat(1, items.shape[1]).view(
            items.shape[0], items.shape[1], -1)
        g_u_v = g_user_embed.repeat(1, items.shape[1]).view(
            items.shape[0], items.shape[1], -1)
        i_v = item_embed
        g_i_v = g_item_embed
        return {
            'context_embedding': context_embedding,
            'prediction': prediction.view(feed_dict['batch_size'], -1),
            'u_v': u_v,
            'i_v': i_v,
            'g_u_v': g_u_v,
            'g_i_v': g_i_v
        }


class LightGCL(GeneralModel, LightGCLBase):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'batch_size',
                      'q', 'temp', 'lambda1', 'lambda2','ssl_temp','ssl_reg','hyper_layers']

    @staticmethod
    def parse_model_args(parser):
        parser = LightGCLBase.parse_model_args(parser)
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def sample_neighbors(self, nodes, adj):
        """采样固定数量的邻居节点"""
        neighbor_dist = adj[nodes]
        # 获取邻居分布的维度
        if isinstance(neighbor_dist, torch.Tensor):
            num_neighbors_available = neighbor_dist.shape[1]
        else:
            # 如果是稀疏矩阵，需要转换为稠密张量
            neighbor_dist = torch.tensor(neighbor_dist.todense()).to(nodes.device)
            num_neighbors_available = neighbor_dist.shape[1]
        
        # 确保概率分布有效
        # 将负值设为0
        neighbor_dist = torch.clamp(neighbor_dist, min=0.0)
        # 为每一行添加一个小的常数，确保概率和大于0
        neighbor_dist = neighbor_dist + 1e-10
        # 归一化每一行，使其成为有效的概率分布
        row_sums = neighbor_dist.sum(dim=1, keepdim=True)
        neighbor_dist = neighbor_dist / row_sums
        
        # 随机采样num_neighbors个邻居
        sampled_neighbors = torch.multinomial(
            neighbor_dist, 
            num_samples=min(self.num_neighbors, num_neighbors_available),
            replacement=True
        )
        return sampled_neighbors

    def forward(self, feed_dict):
        out_dict = LightGCLBase.forward(self, feed_dict)
        # 获取不同层的嵌入用于SSL
        center_embedding = torch.cat([self.encoder.embedding_dict['user_emb'], self.encoder.embedding_dict['item_emb']], dim=0)
        context_embedding = out_dict['context_embedding'] # 需要在encoder中返回
        
        # 计算SSL损失
        ssl_loss = self.ssl_layer_loss(context_embedding, center_embedding, 
                                     feed_dict['user_id'], feed_dict['item_id'])
        out_dict['ssl_loss'] = ssl_loss
        return out_dict

    def ssl_layer_loss(self, current_embedding, previous_embedding, users, items):
        """计算邻居对比损失"""
        # 分离用户和物品嵌入
        current_user_emb, current_item_emb = torch.split(current_embedding, [self.user_num, self.item_num])
        previous_user_emb, previous_item_emb = torch.split(previous_embedding, [self.user_num, self.item_num])
        
        # 获取当前批次的嵌入
        cur_user_emb = current_user_emb[users]
        prev_user_emb = previous_user_emb[users]
        cur_item_emb = current_item_emb[items]
        prev_item_emb = previous_item_emb[items]
        
        # 归一化嵌入
        norm_user_emb1 = F.normalize(cur_user_emb)
        norm_user_emb2 = F.normalize(prev_user_emb)
        norm_all_user_emb = F.normalize(previous_user_emb)
        
        norm_item_emb1 = F.normalize(cur_item_emb)
        norm_item_emb2 = F.normalize(prev_item_emb) 
        norm_all_item_emb = F.normalize(previous_item_emb)
        
        # 计算用户对比损失
        pos_score_user = (norm_user_emb1 * norm_user_emb2).sum(dim=-1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.T)
        
        # 使用 LogSumExp 技巧来提高数值稳定性
        pos_score_user = pos_score_user / self.ssl_temp
        ttl_score_user = ttl_score_user / self.ssl_temp
        
        # 计算 log_sum_exp
        max_score_user = torch.max(ttl_score_user, dim=1, keepdim=True)[0]
        ttl_score_user = max_score_user + torch.log(torch.exp(ttl_score_user - max_score_user).sum(dim=1) + 1e-10)
        ssl_loss_user = (ttl_score_user - pos_score_user).mean()
        
        # 计算物品对比损失
        pos_score_item = (norm_item_emb1 * norm_item_emb2).sum(dim=-1)  # [256, 100]
        
        # 只对当前批次中的物品计算对比损失
        batch_items = items.view(-1)  # 展平物品索引
        unique_items = torch.unique(batch_items)  # 获取唯一的物品索引
        relevant_item_emb = norm_all_item_emb[unique_items]  # 只选择相关的物品嵌入
        
        ttl_score_item = torch.matmul(norm_item_emb1.view(-1, self.emb_size), relevant_item_emb.T)  # [256*100, num_unique_items]
        
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp).view(-1)  # 展平为一维
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1) + 1e-10  # [256*100] 
        
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).mean()
        print(ssl_loss_user, ssl_loss_item)
        return self.ssl_reg * (ssl_loss_user + ssl_loss_item)

    def loss(self, out_dict):
        base_loss = super().loss(out_dict)
        print(base_loss, out_dict['ssl_loss'])
        return base_loss + out_dict['ssl_loss']


class LGCLEncoder(nn.Module):
    def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers, q, temp, lambda1, lambda2, hyper_layers):
        super(LGCLEncoder, self).__init__()

        self.user_count = user_count
        self.item_count = item_count
        self.emb_size = emb_size
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.q = q
        self.temp = temp
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.hyper_layers = hyper_layers
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(
            self.norm_adj).cuda()
        
        # # 添加注意力机制
        # self.W_a = nn.Parameter(torch.empty(emb_size, emb_size))
        # self.a = nn.Parameter(torch.empty(2 * emb_size, 1))
        # self.leakyrelu = nn.LeakyReLU(0.2)
        # # 初始化注意力参数
        # nn.init.xavier_uniform_(self.W_a)
        # nn.init.xavier_uniform_(self.a)

    def attention(self, query, key, value, mask=None):
        # 将 query 和 key 通过线性变换
        query = torch.matmul(query, self.W_a)  # [14682, 64]
        key = torch.matmul(key, self.W_a)      # [14682, 64]
        
        # 计算注意力分数
        # 首先调整维度，添加 batch 维度
        query = query.unsqueeze(0)  # [1, 14682, 64]
        key = key.unsqueeze(0)      # [1, 14682, 64]
        value = value.unsqueeze(0)  # [1, 14682, 64]
        
        # 计算注意力分数
        attn_score = torch.bmm(query, key.transpose(1, 2))  # [1, 14682, 14682]
        attn_score = self.leakyrelu(attn_score)
        
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)
        
        # 应用 softmax
        attn_weights = F.softmax(attn_score, dim=-1)  # [1, 14682, 14682]
        
        # 计算输出
        output = torch.bmm(attn_weights, value)  # [1, 14682, 64]
        
        # 去除添加的 batch 维度
        output = output.squeeze(0)  # [14682, 64]
        
        return output, attn_weights.squeeze(0)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
        })
        return embedding_dict

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    @staticmethod
    def cal_svd_s_v_d(norm_adj, q):
        svd_u, svd_v, svd_d = torch.svd_lowrank(norm_adj, q=q)
        return svd_u, svd_v, svd_d

    def forward(self, users, items):
        # Compute SVD components
        svd_u, svd_s, svd_v = self.cal_svd_s_v_d(
            self.sparse_norm_adj, q=self.q)
        u_mul_s = svd_u @ torch.diag(svd_s)
        v_mul_s = svd_v @ torch.diag(svd_s)
        vt = svd_v.T
        ut = svd_u.T

        # Initialize embeddings for users and items
        E_u_list = [self.embedding_dict['user_emb']]
        E_i_list = [self.embedding_dict['item_emb']]
        G_u_list = []
        G_i_list = []

        for layer in range(1, self.n_layers + 1):
            # GNN propagation with sparse dropout
            Z_u = torch.spmm(
                self.sparse_norm_adj,
                E_i_list[layer - 1]
            )
            Z_i = torch.spmm(
                self.sparse_norm_adj.T,
                E_u_list[layer - 1]
            )

            # # 添加注意力机制
            # Z_u_attn, _ = self.attention(Z_u, Z_u, Z_u)
            # Z_i_attn, _ = self.attention(Z_i, Z_i, Z_i)

            # SVD propagation
            vt_ei = vt @ E_i_list[layer - 1]
            G_u = u_mul_s @ vt_ei
            ut_eu = ut @ E_u_list[layer - 1]
            G_i = v_mul_s @ ut_eu

            # # 结合注意力结果
            # Z_u = Z_u + Z_u_attn
            # Z_i = Z_i + Z_i_attn

            # Aggregate
            E_u_list.append(Z_u)
            E_i_list.append(Z_i)
            G_u_list.append(G_u)
            G_i_list.append(G_i)

        # Sum over all layers
        G_u = sum(G_u_list)
        G_i = sum(G_i_list)
        E_u = sum(E_u_list)
        E_i = sum(E_i_list)

        # Combine GNN and SVD embeddings
        user_all_embeddings = E_u
        item_all_embeddings = E_i

        # Get the embeddings for the specified users and items
        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]
        G_user_embeddings = G_u[users, :]
        G_item_embeddings = G_i[items, :]

        # 修改这里的context_embedding计算
        context_embedding = torch.cat([E_u_list[self.hyper_layers], E_i_list[self.hyper_layers]], dim=0)
        
        out_dict = {
            'user_embeddings': user_embeddings,
            'item_embeddings': item_embeddings,
            'G_user_embeddings': G_user_embeddings,
            'G_item_embeddings': G_item_embeddings,
            'context_embedding': context_embedding
        }
        return out_dict
