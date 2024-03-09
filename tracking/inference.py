import argparse
from configs import cfg
import os
import time
import os.path as osp
from loguru import logger

import dgl
from torchreid.utils import FeatureExtractor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import json
import cv2
import copy
from tqdm import trange

from torchvision import transforms as T
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

import networkx as nx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="./config.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

# for visualization
COLORS = [[39, 188, 221], [167, 72, 214], [82, 67, 198], [76, 198, 232],
          [137, 24, 13], [142, 31, 221], [47, 196, 154], [40, 110, 201],
          [10, 147, 115], [71, 4, 216], [85, 113, 224], [41, 173, 118],
          [52, 172, 237], [80, 237, 164], [175, 164, 65], [70, 53, 178],
          [39, 135, 4], [242, 55, 201], [221, 31, 180], [89, 224, 170],
          [117, 21, 43], [34, 205, 214], [114, 244, 22], [181, 126, 39],
          [127, 17, 69], [102, 12, 211], [26, 178, 127], [198, 67, 249],
          [96, 45, 6], [165, 104, 58]]

def count_human(cfg, graph, file_name, now=None):
    ## Temporal Graph (colored by fID)
    nodes = graph.nodes()
    colors = []
    for n in nodes:
        if graph.nodes[n]['fID'] == now:
            colors.append(1)
        else:
            colors.append(0)
    with open(file_name, 'a') as file:
        file.write(str(colors.count(1)) + ',')
    
def udf_collate_fn(batch):
    return batch

def get_color(idx: int):
    idx = idx * 3
    return (37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

class BaseGraphDataset(Dataset):
    """Class for Graph Dataset."""

    def __init__(self, cfg, seq_names: list, mode: str, feature_extractor, dataset_dir: str):
        self.cfg = cfg
        self.seq_names = seq_names
        self.mode = mode
        self.device = self.cfg.MODEL.DEVICE
        self.feature_extractor = feature_extractor
        self.dataset_dir = dataset_dir

        self._H = []  # homography matrices, H[seq_id][cam_id] => torch.Tensor(3*3)
        self._P = []  # images name pattern, F[seq_id][cam_id] => image path pattern
        self._S = []  # frames annotation in sequences, S[seq_id] => frame based dict (key type: str)
        self._SFI = None  # a (N*2) size tensor, store <seq_id, frame_id>
        self.load_dataset()

        self.frame_range = [self._SFI[0][1], self._SFI[-1][1]]

        self.graph = None # for inference

        self.nodes = [] # use fid as index
        self.load_chunks()

    def load_dataset(self):
        with open(osp.join(self.dataset_dir, 'metainfo.json')) as fp:
            meta_info = json.load(fp)

        if len(self.seq_names) == 1 and self.seq_names[0] == 'all':
            self.seq_names = list(meta_info.keys())

        SFI = []
        for seq_id, name in enumerate(self.seq_names):
            output_path = osp.join(self.dataset_dir, name, 'output', f'{self.cfg.MODEL.DETECTION}_{self.mode}.json')
            with open(output_path, 'r') as fp:
                frames = json.load(fp)
            frames_id = list(map(int, frames.keys()))
            f_idx = torch.tensor(frames_id, dtype=torch.int32).unsqueeze(1)
            s_idx = torch.full_like(f_idx, seq_id)
            SFI.append(torch.hstack([s_idx, f_idx]))
            self._S.append(frames)
            self._H.append(torch.tensor(meta_info[name]['homography']))
            self._P.append([f'{self.dataset_dir}/{name}/output/frames/{{}}_{i}.jpg'
                            for i in range(meta_info[name]['cam_nbr'])])
        self._SFI = torch.vstack(SFI)

    def load_chunks(self):
        num_frames = self._SFI.shape[0]
        sid = 0

        for t in trange(num_frames):
            sid, fid = tuple(map(int, self._SFI[t]))
            frame_images = self.load_images(sid, fid) # {cameras} images with [C, H, W]
            frames = torch.tensor(self._S[sid][str(fid)], dtype=torch.int32)
            n_node = frames.shape[0] # number of detection

            projs = torch.zeros(n_node, 3, dtype=torch.float32)
            (H, W) = self.cfg.FE.INPUT_SIZE
            bdets = torch.zeros(n_node, 3, H, W, dtype=torch.float32) # (N, C, H, W)
            bboxs = torch.zeros(n_node, 4, dtype=torch.float32) # x, y, w, h
            cIDs = torch.zeros(n_node, 1, dtype=torch.int8) # camera ID
            fIDs = torch.zeros(n_node, 1, dtype=torch.int16) # frame ID (timestamp)
            tIDs = torch.zeros(n_node, 1, dtype=torch.int16) # track ID (person ID)
            tIDs_pred = torch.zeros(n_node, 1, dtype=torch.int16) # track ID (person ID) for inference

            for n in range(n_node):
                tid, cid = frames[n, -2:]
                x, y, w, h = frames[n, :4]

                proj = torch.matmul(torch.linalg.inv(self._H[sid][cid]),
                                    torch.t(torch.tensor([x + w / 2, y + h, 1], dtype=torch.float32)))
                
                projs[n] = proj / proj[-1]

                det = frame_images[int(cid)][:, y: y + h, x: x + w]
                det = T.Resize((H, W), antialias=True)(det)
                bdets[n] = det
                bboxs[n] = frames[n, :4]
                cIDs[n] = cid
                fIDs[n] = fid
                tIDs[n] = tid
                tIDs_pred[n] = -1 # default

            if self.cfg.FE.CHOICE == 'CNN':
                # Original CNN re-ID feature extractor
                det_feature = self.feature_extractor(bdets)  # (N, 512)

            nodes_attr = {'cID': cIDs.to(self.device), # [1]
                          'fID': fIDs.to(self.device), # [1]
                          'tID': tIDs.to(self.device), # [1]
                          'tID_pred': tIDs_pred.to(self.device), # [1]
                          'bbox': bboxs.to(self.device), # [4], bbox info. (x, y, w, h)
                          'feat': det_feature.to(self.device), # [512], re-ID(appearance) feature
                          'proj': projs.to(self.device) # [3], geometric position
                         }
            self.nodes.append(nodes_attr)

    def load_images(self, seq_id: int, frame_id: int, tensor=True):
        imgs = []
        for img_path in self._P[seq_id]:
            img = cv2.imread(img_path.format(frame_id))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if tensor:
                img = T.ToTensor()(img)  # (C, H, W), float
            else:
                img = torch.from_numpy(img)
                img = torch.permute(img, (2, 0, 1))  # (C, H, W), uint8
            imgs.append(img)
        return imgs

    def __len__(self):
        return self._SFI.shape[0] # length of frames

    def __getitem__(self, index):
        return self.__getInference__(index)


    def __getInference__(self, index):
        # for spatial graph only in inference stage.
        sid, fid = tuple(map(int, self._SFI[index]))
        frames = torch.tensor(self._S[sid][str(fid)], dtype=torch.int32)
        n_node = frames.shape[0]

        self.graph = dgl.graph(([], []), idtype=torch.int32, device=self.device)
        g = self.graph

        # add new detections(nodes)
        g.add_nodes(n_node, self.nodes[index])

        # add edge
        u, v = [], []
        for n in range(g.num_nodes()):
            u += [n] * g.num_nodes()
            v += list(range(g.num_nodes()))
        g.add_edges(u, v)

        g_cID = g.ndata['cID']
        _from = g.edges()[0].type(torch.long)
        _to = g.edges()[1].type(torch.long)
        li = torch.where(g_cID[_from]==g_cID[_to])[0]
        assert len(li) != 0
        g.remove_edges(list(li))

        node_feature = g.ndata['feat']
        projs = g.ndata['proj']
        u = g.edges()[0].type(torch.long)
        v = g.edges()[1].type(torch.long)

        edge_feature = torch.vstack((
            torch.pairwise_distance(node_feature[u], node_feature[v]).to(self.device),
            1 - torch.cosine_similarity(node_feature[u], node_feature[v]).to(self.device),
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=1).to(self.device),
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=2).to(self.device)
        )).T  # (E, 4)
        g.edata['embed'] = edge_feature

        return self.graph, node_feature, edge_feature


class MPN(nn.Module):
    """Message Passing Neural Network."""

    def __init__(self, cfg, ckpt=None):
        super(MPN, self).__init__()
        # Learnable MLP message encoders
        self.node_msg_encoder = nn.Sequential(
            nn.Linear(38, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.edge_msg_encoder = nn.Sequential(
            nn.Linear(70, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
            nn.ReLU()
        )
        self.to(cfg.MODEL.DEVICE)
        if ckpt is not None:
            self.load_state_dict(ckpt)

    def message_udf(self, edges):
        """ User define function message function
        """
        edge_msg = self.edge_msg_encoder(
            torch.cat((edges.dst['x'], edges.src['x'], edges.data['x']), 1)
        )
        self.em = edge_msg

        # Aggregate adjacent node feature(32) & edge feature to itself (6) => AGgregateEdgeMessage
        node_msg = self.node_msg_encoder(
            torch.cat((edges.dst['x'], edge_msg), 1)
        )
        return {'agem': node_msg}

    def reduce_udf(self, nodes):
        """ Udf aggregating function
        """
        return {'nm': nodes.mailbox['agem'].sum(dim=1)}

    def forward(self, graph, x_node, x_edge):
        with graph.local_scope():
            graph.ndata['x'] = x_node
            graph.edata['x'] = x_edge

            graph.update_all(message_func=self.message_udf, reduce_func=self.reduce_udf)
            return graph.ndata['nm'], self.em
        
class NodeFeatureEncoder(nn.Module):
    def __init__(self, cfg, ckpt=None, in_dim=None):
        super(NodeFeatureEncoder, self).__init__()
        if in_dim is None:
            in_dim = 516 if cfg.SOLVER.TYPE == 'TG' else 512

        self.layer = nn.Sequential(
            nn.Linear(in_dim, 128), # 516
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.to(cfg.MODEL.DEVICE)
        if ckpt is not None:
            self.load_state_dict(ckpt)

    def forward(self, x):
        return self.layer(x)

class EdgeFeatureEncoder(nn.Module):
    def __init__(self, cfg, ckpt=None, in_dim=None):
        super(EdgeFeatureEncoder, self).__init__()
        if in_dim is None:
            # 6: add velocity feature
            in_dim = 6 if cfg.SOLVER.TYPE == 'TG' else 4
        self.layer = nn.Sequential(
            nn.Linear(in_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU()
        )
        self.to(cfg.MODEL.DEVICE)
        if ckpt is not None:
            self.load_state_dict(ckpt)

    def forward(self, x):
        return self.layer(x)

class EdgePredictor(nn.Module):
    def __init__(self, cfg, ckpt=None):
        super(EdgePredictor, self).__init__()
        self.pred = nn.Sequential(
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        self.to(cfg.MODEL.DEVICE)
        if ckpt is not None:
            self.load_state_dict(ckpt)

    def forward(self, x_edge):
        return self.pred(x_edge)
    

class Tracklet():
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        self.outpu_dir = output_dir
        self.graph = None
        self.y_pred = None
        self.newID = 0

    def inference(self, time, graph, y_pred, mode):
        self.time = time
        self.graph = graph
        self.y_pred = y_pred
        self.graph.edata['y_pred'] = self.y_pred

        graph = None
        if mode == 'SG':
            self.remove_edge()
            self.postprocessing_sg()
            graph = self.aggregating_sg()

        if mode == 'TG':
            self.remove_edge()
            self.postprocessing_tg()
            self.assign_ID()
            self.write_infer_file()
            graph = self.aggregating_tg()

        return graph


    def remove_edge(self):

        edge_thresh = self.cfg.TEST.EDGE_THRESH
       
        g = self.graph
        orig_edge = g.num_edges()
        li = torch.where(self.y_pred < edge_thresh)[0]
        if len(li) > 0:
            g.remove_edges(list(li))

        # remove multi-edge(more than one edge b/t two nodes)
        edge_set = set()
        rm_li = []
        _from = g.edges()[0].type(torch.long)
        _to = g.edges()[1].type(torch.long)
        for i in range(g.num_edges()):
            f = _from[i].item()
            t = _to[i].item()
            if (f, t) in edge_set:
                rm_li.append(i)
            else:
                edge_set.add((f, t))
                edge_set.add((t, f))
        
        g.remove_edges(rm_li)

    def postprocessing_sg(self):
        # 1. Eq.(11): degree(v) < C-1   -> splitting_1()
        # 2. Eq.(10): |V(H)| < C        -> splitting_2()
        rm_len = 0

        rm_len += self.splitting_1()  ##degree

        rml, again = self.splitting_2()  ## number of connected components
        rm_len += rml

    def toSimpleGraph(self, mg):
        G = nx.Graph()
        for u, v, data in mg.edges(data=True):
            w = data['y_pred'] if 'y_pred' in data else 1.0
            G.add_edge(u, v, weight=w)
        return G

    def findEdgebyNode(self, u, v):
        _from = self.graph.edges()[0].type(torch.long)
        _to = self.graph.edges()[1].type(torch.long)
        fli = torch.where(_from == u)[0].tolist()
        tli = torch.where(_to == v)[0].tolist()
        eid = list(set(fli).intersection(tli))

        if len(eid) == 0:
            fli = torch.where(_from == v)[0].tolist()
            tli = torch.where(_to == u)[0].tolist()
            eid = list(set(fli).intersection(tli))

        return eid[0]

    def findMinEdge(self, edges, g_ypred):
        min_edge = -1
        min_pred = 1.1
        rm_f, rm_t = -1, -1
        for f, t in edges:
            eid = self.findEdgebyNode(f, t)
            if g_ypred[eid] < min_pred:
                min_pred = g_ypred[eid]
                min_edge = eid
                rm_f = f
                rm_t = t
        return min_edge, rm_f, rm_t

    def splitting_1(self):
        nxg = dgl.to_networkx(self.graph.cpu(),
                              edge_attrs=['y_pred']).to_undirected()
        nxg = self.toSimpleGraph(nxg)

        g_ypred = self.graph.edata['y_pred']
        rm_li = []
        for i, cc in enumerate(nx.connected_components(nxg)):
            ccli = list(cc)
            sg = nxg.subgraph(cc)
            flows = [d for n, d in sg.degree(ccli)]
            violate = torch.where(
                torch.tensor(flows) > self.cfg.DATASET.CAMS - 1)[0]
            while len(violate) > 0:
                bridge = list(nx.bridges(sg))
                if len(bridge) == 1:
                    rm_f, rm_t = bridge[0]
                    rm_li.append(self.findEdgebyNode(rm_f, rm_t))
                elif len(bridge) > 1:  # more than one bridge
                    rm_eid, rm_f, rm_t = self.findMinEdge(bridge, g_ypred)
                    rm_li.append(rm_eid)
                else:  # no bridge
                    rm_eid, rm_f, rm_t = self.findMinEdge(sg.edges(), g_ypred)
                    rm_li.append(rm_eid)
                sg = nx.Graph(sg)
                sg.remove_edge(rm_f, rm_t)
                flows = [d for n, d in sg.degree(ccli)]
                violate = torch.where(
                    torch.tensor(flows) > self.cfg.DATASET.CAMS - 1)[0]

        self.graph.remove_edges(rm_li)
        
        return len(rm_li)

    def splitting_2(self):
        nxg = dgl.to_networkx(self.graph.cpu(),
                              edge_attrs=['y_pred']).to_undirected()
        nxg = self.toSimpleGraph(nxg)

        g_ypred = self.graph.edata['y_pred']
        rm_li = []
        again = False
        for i, cc in enumerate(nx.connected_components(nxg)):
            ccli = list(cc)
            sg = nxg.subgraph(cc)
            violate = True if len(ccli) > self.cfg.DATASET.CAMS else False
            while violate:  # violate condition
                bridge = list(nx.bridges(sg))
                if len(bridge) == 1:
                    rm_f, rm_t = bridge[0]
                    rm_li.append(self.findEdgebyNode(rm_f, rm_t))
                elif len(bridge) > 1:
                    rm_eid, rm_f, rm_t = self.findMinEdge(bridge, g_ypred)
                    rm_li.append(rm_eid)
                else:  # no bridge
                    rm_eid, rm_f, rm_t = self.findMinEdge(sg.edges(), g_ypred)
                    rm_li.append(rm_eid)

                sg = nx.Graph(sg)
                sg.remove_edge(rm_f, rm_t)
                cnt = 0
                error = False
                for i, cc in enumerate(nx.connected_components(sg)):
                    cnt += 1
                    if len(list(cc)) > self.cfg.DATASET.CAMS:
                        error = True
                if cnt > 1:  # successfully break into two c.c
                    if error:  # but still violate, do whole function again
                        again = True
                    break
        self.graph.remove_edges(rm_li)
        
        return len(rm_li), again

    def aggregating_sg(self):
        nxg = dgl.to_networkx(self.graph.cpu(),
                              node_attrs=['tID']).to_undirected()

        g_tID = self.graph.ndata['tID']
        g_cID = self.graph.ndata['cID']
        g_feat = self.graph.ndata['feat']
        g_proj = self.graph.ndata['proj']
        g_bbox = self.graph.ndata['bbox']

        n_node = 0
        for i, cc in enumerate(nx.connected_components(nxg)):
            n_node += 1
        fIDs = torch.zeros(n_node, 1, dtype=torch.int16)
        feats = torch.zeros(n_node, 512, dtype=torch.float32)
        projs = torch.zeros(n_node, 3, dtype=torch.float32)
        velocitys = torch.zeros(n_node, 2, dtype=torch.float32)
        tIDs_pred = torch.zeros(n_node, 1, dtype=torch.int16)
        self.bboxs = []

        for i, cc in enumerate(nx.connected_components(nxg)):
            ccli = list(cc)
            fIDs[i] = self.time + self.cfg.TEST.FRAME_START
            feats[i] = torch.mean(g_feat[ccli], 0)
            projs[i] = torch.mean(g_proj[ccli], 0)
            tIDs_pred[i] = -1

            # save bbox and cID for inference
            tmp_li = []
            for node in ccli:
                tmp_li.append([
                    g_cID[node].item(),
                    [g_bbox[node][i].item() for i in range(4)]
                ])
            self.bboxs.append(tmp_li)

        nodes_attr = {
            'fID': fIDs.to(self.device),  # current time
            'feat': feats.to(self.device),  # mean feature of all nodes
            'proj': projs.to(self.device),  # mean projection of all nodes
            'velocity':
            velocitys.to(self.device),  # init v of spatial node to 0
            'tID_pred': tIDs_pred.to(self.device)  # initialize for inference
        }

        # create SG(node-only)
        sg = dgl.graph(([], []), idtype=torch.int32, device=self.device)
        sg.add_nodes(n_node, nodes_attr)

        return sg

    def postprocessing_tg(self):
        # Eq.(12): degree(v) < M-1
        nxg = dgl.to_networkx(self.graph.cpu(),
                              edge_attrs=['y_pred']).to_undirected()
        nxg = self.toSimpleGraph(nxg)

        g_ypred = self.graph.edata['y_pred']
        rm_li = []

        for i, cc in enumerate(nx.connected_components(nxg)):
            ccli = list(cc)
            sg = nxg.subgraph(cc)
            flows = [d for n, d in sg.degree(ccli)]
            violate = torch.where(torch.tensor(flows) > 1)[0]
            while len(violate) > 0:  # solve many-to-one violation
                rm_eid, rm_f, rm_t = self.findMinEdge(sg.edges(), g_ypred)
                rm_li.append(rm_eid)

                sg = nx.Graph(sg)
                sg.remove_edge(rm_f, rm_t)
                flows = [d for n, d in sg.degree(ccli)]
                violate = torch.where(torch.tensor(flows) > 1)[0]
        self.graph.remove_edges(rm_li)

    def assign_ID(self):
        g_tID_pred = self.graph.ndata['tID_pred']
        nxg = dgl.to_networkx(
            self.graph.cpu(), node_attrs=['fID'], edge_attrs=['y_pred']
        ).to_undirected()  # node_attrs=['tID_pred'], edge_attrs=['y_pred']
        count_human(self.cfg,
                       nxg,
                       os.path.join(self.outpu_dir, 'count.txt'),
                       now=self.time + self.cfg.TEST.FRAME_START)
        g_fID = self.graph.ndata['fID']
        for i, cc in enumerate(nx.connected_components(nxg)):
            nodes = list(cc)
            labeled_nodes = torch.where(
                g_tID_pred[nodes] != -1)[0]  # return list of index
            if len(labeled_nodes) > 0:
                label = copy.deepcopy(
                    g_tID_pred[nodes[labeled_nodes[0]]])
                self.graph.ndata['tID_pred'][nodes] = label
            else:
                label = self.newID
                self.graph.ndata['tID_pred'][nodes] = label
                self.newID += 1

    def write_infer_file(self):
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        if self.time == 0:
            return
        attr = [
            'frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height',
            'conf', 'x', 'y', 'z'
        ]
        
        frame = f'0000{(self.time + self.cfg.TEST.FRAME_START)*5:04d}'
        
        current_fID = self.time + self.cfg.TEST.FRAME_START

        g_fID = self.graph.ndata['fID']
        g_tID = self.graph.ndata['tID_pred']

        dfs = []
        for _ in range(self.cfg.DATASET.CAMS):
            dfs.append(pd.DataFrame(columns=attr))

        for n in range(self.graph.num_nodes()):
            if g_fID[n] != current_fID:  # online method, only preocess current nodes
                continue
            for i in range(len(self.bboxs[n])):
                x, y, w, h = self.bboxs[n][i][1]
                c = self.bboxs[n][i][0]
                dfs[c] = dfs[c]._append(
                    {
                        'frame': frame,
                        'id': g_tID[n].cpu().item(),
                        'bb_left': x,
                        'bb_top': y,
                        'bb_width': w,
                        'bb_height': h,
                        'conf': 1,
                        'x': -1,
                        'y': -1,
                        'z': -1
                    },
                    ignore_index=True)
        for c in range(self.cfg.DATASET.CAMS):
            dfs[c].to_csv(os.path.join(self.outpu_dir, f'c{c}.txt'),
                          header=None,
                          index=None,
                          sep=',',
                          mode='a')

    def aggregating_tg(self):
        nxg = dgl.to_networkx(self.graph.cpu()).to_undirected()

        g_feat = self.graph.ndata['feat']
        g_proj = self.graph.ndata['proj']
        g_fID = self.graph.ndata['fID']
        g_tIDpred = self.graph.ndata['tID_pred']

        n_node = 0
        for i, cc in enumerate(nx.connected_components(nxg)):
            n_node += 1
        fIDs = torch.zeros(n_node, 1, dtype=torch.int16)
        feats = torch.zeros(n_node, 512, dtype=torch.float32)
        projs = torch.zeros(n_node, 3, dtype=torch.float32)
        tIDs_pred = torch.zeros(n_node, 1, dtype=torch.int16)
        velocitys = torch.zeros(n_node, 2, dtype=torch.float32)

        for i, cc in enumerate(nx.connected_components(nxg)):
            ccli = list(cc)
            if len(ccli) == 1:  # no match now
                fIDs[i] = g_fID[ccli[0]]
            else:
                fIDs[
                    i] = self.time + self.cfg.TEST.FRAME_START  # successful match, update time
            feats[i] = torch.mean(g_feat[ccli], 0)
            projs[i] = torch.mean(g_proj[ccli], 0)
            tIDs_pred[i] = g_tIDpred[ccli][0].item()
            if len(ccli) == 2:
                _pre, _now = -1, -1
                if g_fID[ccli][0] < g_fID[ccli][1]:
                    _pre, _now = 0, 1
                elif g_fID[ccli][0] > g_fID[ccli][1]:
                    _pre, _now = 1, 0
                if _pre != -1 and _now != -1:
                    velocitys[
                        i] = g_proj[ccli][_now][:2] - g_proj[ccli][_pre][:2]

        nodes_attr = {
            'fID': fIDs.to(self.device),  # current time
            'feat': feats.to(self.device),  # mean feature of all nodes
            'proj': projs.to(self.device),  # mean projection of all nodes
            'velocity': velocitys.to(self.device),  # velocity
            'tID_pred': tIDs_pred.to(self.device)  # initialize for inference
        }

        # create pre-TG(node-only)
        pre_tg = dgl.graph(([], []), idtype=torch.int32, device=self.device)
        pre_tg.add_nodes(n_node, nodes_attr)

        return pre_tg

    def visualize(self, time, vis_output_dir):
        frame = time + self.cfg.TEST.FRAME_START
        output_dir = os.path.join(vis_output_dir, 'frames')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            for c in range(self.cfg.DATASET.CAMS):
                os.mkdir(os.path.join(output_dir, f'c{c}'))

        g_fID = self.graph.ndata['fID']
        g_tID = self.graph.ndata['tID_pred']
        g_proj = self.graph.ndata['proj']

        c = self.cfg.DATASET

        cam_nodes = []
        for _ in range(c.CAMS):
            cam_nodes.append([])
        for n in range(self.graph.num_nodes()):
            if g_fID[n] != frame:
                continue
            tID = g_tID[n]
            proj = g_proj[n]
            for i in range(len(self.bboxs[n])):
                x, y, w, h = self.bboxs[n][i][1]
                ca = self.bboxs[n][i][0]
                cam_nodes[ca].append([x, y, w, h, tID, proj])

        bird_view = np.zeros((1080, 1920, 3))
        
        for cam in range(c.CAMS):
            frame_img = os.path.join(c.DIR, c.NAME, c.SEQUENCE[0],
                                     'output/frames', f'{frame}_{cam}.jpg')
            img = cv2.imread(frame_img)
            for b in range(len(cam_nodes[cam])):
                bbox = cam_nodes[cam][b][:4]
                tID_pred = cam_nodes[cam][b][4]
                proj = cam_nodes[cam][b][5]

                # bbox and label with color
                color = (COLORS[tID_pred.item() % 30])
                cv2.rectangle(
                    img, (int(bbox[0]), int(bbox[1])),
                    (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])),
                    color, 5)  #2
                cv2.rectangle(img, (int(bbox[0]) - 5, int(bbox[1]) - 40),
                              (int(bbox[0]) + 60, int(bbox[1])), color, -1)  #2
                cv2.putText(img, f'{tID_pred.item()}',
                            (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2,
                            cv2.LINE_AA)

                # bird view visualization
                bird_view = cv2.circle(bird_view, ((int(proj[0])+1200)//2, int(proj[1])-140), 22, color, -1)
            
            cv2.imwrite(os.path.join(output_dir, f'c{cam}/{frame}.jpg'), img)
        

        

        # bird view visualization
        path = os.path.join(output_dir, 'bird_view')
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(os.path.join(path, f'{frame}.jpg'), bird_view)


class Tracker:

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device=self.cfg.MODEL.DEVICE
        self.min_loss = 1e8

        if self.cfg.FE.CHOICE == 'CNN':
            # Default ReID model, OS-Net here
            self.feature_extractor = FeatureExtractor(
                model_name='osnet_ain_x1_0',
                model_path='./osnet_ain_ms_d_c.pth.tar',
                device=self.device
            )

        self.node_feature_encoder = NodeFeatureEncoder(self.cfg)
        self.edge_feature_encoder = EdgeFeatureEncoder(self.cfg)
        self.mpn = MPN(self.cfg)
        self.predictor = EdgePredictor(self.cfg)
    
        self.output_dir = osp.join(self.cfg.OUTPUT.INFERENCE_DIR, f'test-{self.cfg.DATASET.NAME}-{self.cfg.DATASET.SEQUENCE[0]}-{int(time.time())}')
        self.tracklet = Tracklet(self.cfg, self.output_dir)
        

        make_dir(self.output_dir)
        logger.add(f'{self.output_dir}/log.txt')
        logger.info(f"Detection: {self.cfg.MODEL.DETECTION}")
    
    def load_dataset(self):
        dataset_name = self.cfg.DATASET.NAME

        dataset = BaseGraphDataset(self.cfg, self.cfg.DATASET.SEQUENCE, 'test', self.feature_extractor, osp.join(self.cfg.DATASET.DIR, dataset_name))
        
        return dataset

    def load_param(self, mode):
        if mode == 'test':
            ckpt = torch.load(self.cfg.TEST.CKPT_FILE_SG)
            self.SG = {'node_feature_encoder': NodeFeatureEncoder(self.cfg, in_dim=512),
                       'edge_feature_encoder': EdgeFeatureEncoder(self.cfg, in_dim=4),
                       'mpn': MPN(self.cfg),
                       'predictor': EdgePredictor(self.cfg)}
            self.SG['node_feature_encoder'].load_state_dict(ckpt['node_feature_encoder'])
            self.SG['edge_feature_encoder'].load_state_dict(ckpt['edge_feature_encoder'])
            self.SG['mpn'].load_state_dict(ckpt['mpn'])
            self.SG['predictor'].load_state_dict(ckpt['predictor'])
            logger.info(f'Load Spatial Graph param from {self.cfg.TEST.CKPT_FILE_SG}')

            ckpt = torch.load(self.cfg.TEST.CKPT_FILE_TG)
            self.TG = {'node_feature_encoder': NodeFeatureEncoder(self.cfg, in_dim=516),
                       'edge_feature_encoder': EdgeFeatureEncoder(self.cfg, in_dim=6),
                       'mpn': MPN(self.cfg),
                       'predictor': EdgePredictor(self.cfg)}
            self.TG['node_feature_encoder'].load_state_dict(ckpt['node_feature_encoder'])
            self.TG['edge_feature_encoder'].load_state_dict(ckpt['edge_feature_encoder'])
            self.TG['mpn'].load_state_dict(ckpt['mpn'])
            self.TG['predictor'].load_state_dict(ckpt['predictor'])
            logger.info(f'Load Temporal Graph param from {self.cfg.TEST.CKPT_FILE_TG}')

        return ckpt

    def test(self):
        ckpt = self.load_param(self.cfg.MODEL.MODE)

        visualize_dir = None
        if self.cfg.OUTPUT.VISUALIZE:
            visualize_dir = osp.join(self.output_dir, 'visualize')
            make_dir(visualize_dir)

        # test_dataset = self.load_dataset()[0]
        test_dataset = self.load_dataset()
        test_loader = DataLoader(test_dataset, 1, collate_fn=udf_collate_fn)
        self._test_one_epoch(test_loader, ckpt['L'], visualize_dir) # generate inference file

    @torch.no_grad()
    def _test_one_epoch(self, dataloader, max_passing_steps: int, visualize_output_dir=None):
        pre_TG = None
        for i, data in enumerate(dataloader):
            """ Spatial Graph
                Input: detection from each camera at current frame
                Output: A aggregated graph with nodes only
            """
            for graph, node_feature, edge_feature in data:
                x_node = self.SG['node_feature_encoder'](node_feature)
                x_edge = self.SG['edge_feature_encoder'](edge_feature)
                for _ in range(max_passing_steps):
                    x_node, x_edge = self.SG['mpn'](graph, x_node, x_edge)
                y_pred = self.SG['predictor'](x_edge)
                SG = self.tracklet.inference(i, graph, y_pred.flatten(), 'SG') # post-processing & graph reconfiguration
                if pre_TG is None:
                    pre_TG = SG # t=0

            if self.cfg.OUTPUT.LOG:
                logger.opt(colors=True).info(f'<fg 255,204,0>Iteration {i}: Spatial Graph done.'+
                                        f'SG: {SG.num_nodes()} nodes and {SG.num_edges()} edges.</fg 255,204,0>')


            """ Temporal Graph
                Input: SG at time i, TG at time i-1
                Output: inference result at time i
            """
            if i > 0:
                # Add edge (both input graphs are node-only)
                TG, node_feature, edge_feature = self.reconfiguration(pre_TG, SG)

                # Run Temporal Graph
                x_node = self.TG['node_feature_encoder'](node_feature)
                x_edge = self.TG['edge_feature_encoder'](edge_feature)
                for _ in range(max_passing_steps):
                    x_node, x_edge = self.TG['mpn'](TG, x_node, x_edge) # node: 32D, edge: 6D

                y_pred = self.TG['predictor'](x_edge)

                # Post-processing
                TG = self.tracklet.inference(i, TG, y_pred.flatten(), 'TG') # post-processing
                pre_TG = TG

                if self.cfg.OUTPUT.VISUALIZE:
                    self.tracklet.visualize(i, visualize_output_dir)

                if self.cfg.OUTPUT.LOG:
                    logger.opt(colors=True).info(f'<fg 255,204,0>Iteration {i}: Temporal Graph done.'+
                                            f'TG: {TG.num_nodes()} nodes and {TG.num_edges()} edges.</fg 255,204,0>')
            if self.cfg.OUTPUT.LOG:
                logger.opt(colors=True).info(f'<fg 255,153,51>【Finished inference iteration {i}/{len(dataloader)-1}】</fg 255,153,51>\n')

    def reconfiguration(self, pre_TG, SG):
        TG = dgl.graph(([], []), idtype=torch.int32, device=self.device)
        n_node_preTG = pre_TG.num_nodes()
        n_node_SG = SG.num_nodes()
        TG.add_nodes(n_node_SG, SG.ndata)
        TG.add_nodes(n_node_preTG, pre_TG.ndata)

        g_fID = TG.ndata['fID']
        _now = int(max(g_fID))

        # remove old node
        li = torch.where(g_fID < _now - 2)[0]
        TG.remove_nodes(list(li))

        # add edge (revisied ver., don't connect past two nodes)
        _from, _to = [], []
        for n1 in range(TG.num_nodes()):
            if g_fID[n1] != _now:
                continue
            for n2 in range(TG.num_nodes()):
                if g_fID[n2] == _now:
                    continue
                _from.append(n1)
                _to.append(n2)
        TG.add_edges(_from + _to, _to + _from)

        g_fID = TG.ndata['fID']
        reid_feature = TG.ndata['feat']
        projs = TG.ndata['proj']
        velocitys = TG.ndata['velocity']
        g_cID = torch.ones(g_fID.shape).cuda()
        node_feature = torch.cat((reid_feature, projs, g_cID), 1) # 516 d

        u = TG.edges()[0].type(torch.long)
        v = TG.edges()[1].type(torch.long)

        edge_feature = torch.vstack((
            torch.pairwise_distance(reid_feature[u], reid_feature[v]).to(self.device),
            1 - torch.cosine_similarity(reid_feature[u], reid_feature[v]).to(self.device),
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=1).to(self.device),
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=2).to(self.device),
            torch.pairwise_distance(velocitys[u, :2], velocitys[v, :2], p=1).to(self.device),
            torch.pairwise_distance(velocitys[u, :2], velocitys[v, :2], p=2).to(self.device),
        )).T
        TG.edata['embed'] = edge_feature

        return TG, node_feature, edge_feature

if __name__ == "__main__":
    parse_args()
    inference = Tracker(cfg) 
    inference.test()
