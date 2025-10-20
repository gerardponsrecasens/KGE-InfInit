from ..utils import *
from copy import deepcopy as dcopy


class KnowledgeGraph():
    def __init__(self, args):
        self.args = args

        self.num_ent, self.num_rel = 0, 0
        self.entity2id, self.id2entity, self.relation2id, self.id2relation = dict(), dict(), dict(), dict()
        self.relation2inv = dict()

        self.snapshots = {i: SnapShot(self.args) for i in range(int(self.args.snapshot_num))}
        self.load_data()

    def load_data(self):
        '''
        load data from all snapshot file
        '''
        sr2o_all = dict()
        train_all, valid_all, test_all = [], [], []
        for ss_id in range(int(self.args.snapshot_num)):
            self.new_entities = set()
            '''load facts'''
            train_facts = load_fact(self.args.data_path + str(ss_id) + '/' + 'train.txt')
            test_facts = load_fact(self.args.data_path + str(ss_id) + '/' + 'test.txt')
            valid_facts = load_fact(self.args.data_path + str(ss_id) + '/' + 'test.txt')

            '''extract entities & relations from facts'''
            self.expand_entity_relation(train_facts)
            self.expand_entity_relation(valid_facts)
            self.expand_entity_relation(test_facts)

            '''read train/test/valid data'''
            train = self.fact2id(train_facts)
            valid = self.fact2id(valid_facts, order=True)
            test = self.fact2id(test_facts, order=True)

            '''
            Get edge_index and edge_type for GCN"
                edge_index = [[s_1, s_2, ... s_n],[o_1, o_2, ..., o_n]]
                edge_type = [r_1, r_2, ..., r_n]
            '''
            edge_s, edge_r, edge_o = [], [], []
            edge_s, edge_o, edge_r = self.expand_kg(train, 'train', edge_s, edge_o, edge_r, sr2o_all)
            edge_s, edge_o, edge_r = self.expand_kg(valid, 'valid', edge_s, edge_o, edge_r, sr2o_all)
            edge_s, edge_o, edge_r = self.expand_kg(test, 'test', edge_s, edge_o, edge_r, sr2o_all)

            '''prepare data for 're-training' model'''
            train_all += train
            valid_all += valid
            test_all += test

            '''store this snapshot'''
            self.store_snapshot(ss_id, train, train_all, test, test_all, valid, valid_all, edge_s, edge_o, edge_r, sr2o_all)
            self.new_entities.clear()

    def expand_entity_relation(self, facts):
        '''extract entities and relations from new facts'''
        for (s, r, o) in facts:
            '''extract entities'''
            if s not in self.entity2id.keys():
                self.entity2id[s] = self.num_ent
                self.num_ent += 1
            if o not in self.entity2id.keys():
                self.entity2id[o] = self.num_ent
                self.num_ent += 1

            '''extract relations'''
            if r not in self.relation2id.keys():
                self.relation2id[r] = self.num_rel
                self.relation2id[r + '_inv'] = self.num_rel + 1
                self.relation2inv[self.num_rel] = self.num_rel + 1
                self.relation2inv[self.num_rel + 1] = self.num_rel
                self.num_rel += 2

    def fact2id(self, facts, order=False):
        '''(s name, r name, o name)-->(s id, r id, o id)'''
        fact_id = []
        if order:
            i = 0
            while len(fact_id) < len(facts):
                for (s, r, o) in facts:
                    if self.relation2id[r] == i:
                        fact_id.append((self.entity2id[s], self.relation2id[r], self.entity2id[o]))
                i = i + 2
        else:
            for (s, r, o) in facts:
                fact_id.append((self.entity2id[s], self.relation2id[r], self.entity2id[o]))
        return fact_id

    def expand_kg(self, facts, split, edge_s, edge_o, edge_r, sr2o_all):
        '''expand edge_index, edge_type (for GCN) and sr2o (to filter golden facts)'''
        def add_key2val(dict, key, val):
            '''add {key: value} to dict'''
            if key not in dict.keys():
                dict[key] = set()
            dict[key].add(val)

        for (h, r, t) in facts:
            self.new_entities.add(h)
            self.new_entities.add(t)
            if split == 'train':
                '''edge_index'''
                edge_s.append(h)
                edge_r.append(r)
                edge_o.append(t)
            '''sr2o'''
            add_key2val(sr2o_all, (h, r), t)
            add_key2val(sr2o_all, (t, self.relation2inv[r]), h)
        return edge_s, edge_o, edge_r

    def store_snapshot(self, ss_id, train_new, train_all, test, test_all, valid, valid_all, edge_s, edge_o, edge_r, sr2o_all):
        '''store snapshot data'''
        self.snapshots[ss_id].num_ent = dcopy(self.num_ent)
        self.snapshots[ss_id].num_rel = dcopy(self.num_rel)

        '''train, valid and test data'''
        self.snapshots[ss_id].train_new = dcopy(train_new)
        self.snapshots[ss_id].train_all = dcopy(train_all)
        self.snapshots[ss_id].test = dcopy(test)
        self.snapshots[ss_id].valid = dcopy(valid)
        self.snapshots[ss_id].valid_all = dcopy(valid_all)
        self.snapshots[ss_id].test_all = dcopy(test_all)

        '''edge_index, edge_type (for GCN of MEAN and LAN)'''
        self.snapshots[ss_id].edge_s = dcopy(edge_s)
        self.snapshots[ss_id].edge_r = dcopy(edge_r)
        self.snapshots[ss_id].edge_o = dcopy(edge_o)

        '''sr2o (to filter golden facts)'''
        self.snapshots[ss_id].sr2o_all = dcopy(sr2o_all)
        self.snapshots[ss_id].edge_index = build_edge_index(edge_s, edge_o).to(self.args.device)
        self.snapshots[ss_id].edge_type = torch.cat(
            [torch.LongTensor(edge_r), torch.LongTensor(edge_r) + 1]).to(self.args.device)
        self.snapshots[ss_id].new_entities = dcopy(list(self.new_entities))

        if self.args.lifelong_name in ['LAN', 'MEAN']:
            self.snapshots[ss_id].ent2neigh, self.snapshots[ss_id].edge_index_sample, self.snapshots[ss_id].edge_type_sample, self.snapshots[ss_id].ent_neigh_num = self.snapshots[ss_id].sample_neighbor()


class SnapShot():
    def __init__(self, args):
        self.args = args
        self.num_ent, self.num_rel = 0, 0
        self.train_new, self.train_all, self.test, self.valid, self.valid_all, self.test_all = list(), list(), list(), list(), list(), list()
        self.edge_s, self.edge_r, self.edge_o = [], [], []
        self.sr2o_all = dict()
        self.edge_index, self.edge_type = None, None
        self.new_entities = []

    def sample_neighbor(self):
        '''sample neighbor for MEAN or LAN'''
        num = 64
        res = []
        triples = self.train_new
        ent2triples = {i:list() for i in range(self.num_ent)}
        edge_index_sample, edge_type_sample = [], []
        ent_neigh_num = torch.zeros(self.num_ent).to(self.args.device)
        for triple in triples:
            h, r, t = triple
            ent2triples[h].append((h, r, t))
            ent2triples[t].append((t, r+1, h))
        for ent in range(self.num_ent):
            ent2triples[ent].append((ent, self.num_rel, ent))
            if len(ent2triples[ent]) > num:
                ent_neigh_num[ent] = num
                samples = [ent2triples[ent][i] for i in np.random.choice(range(len(ent2triples[ent])), num, replace=False)]
            else:
                samples = ent2triples[ent]
                ent_neigh_num[ent] = len(ent2triples[ent])
                for i in range(num - len(ent2triples[ent])):
                    samples.append((self.num_ent, self.num_rel+1, self.num_ent))
            res.append(samples)
            for hrt in samples:
                h, r, t = hrt
                if r == self.num_rel+1:
                    pass
                else:
                    edge_index_sample.append([ent, t])
                    edge_type_sample.append(r)

        return torch.LongTensor(res).to(self.args.device), torch.LongTensor(edge_index_sample).to(self.args.device).t(), torch.LongTensor(edge_type_sample).to(self.args.device), ent_neigh_num




class KnowledgeGraphIncDE():
    def __init__(self, args) -> None:
        self.args = args
        self.num_ent, self.num_rel = 0, 0
        self.entity2id, self.id2entity, self.relation2id, self.id2relation = {}, {}, {}, {}
        self.relationid2invid = {}
        self.snapshots = {i: SnapshotIncDE(self.args) for i in range(int(self.args.snapshot_num))}
        """ use or not """
        if self.args.use_multi_layers and self.args.first_training:
            self.generate_layers(args)
        self.load_data()

    def ordered_by_edges(self, new_ordered_train_data, ss_id):
        """ Intra-hierarchical sorting: Sort triples by the betweenness centrality of the edges """
        ori_len = len(new_ordered_train_data)
        train_edges_betweenness_path = self.args.data_path + str(ss_id) + "/" + "train_edges_betweenness.txt"
        train_edges_betweenness_dict = dict()
        with open(train_edges_betweenness_path, "r", encoding="utf-8") as rf:
            for line in rf.readlines():
                line_list = line.strip().split("\t")
                node1, node2, value = int(line_list[0]), int(line_list[1]), float(line_list[2])
                train_edges_betweenness_dict[(node1, node2)] = value
                train_edges_betweenness_dict[(node2, node1)] = value
        tmp_ordered_train_data = list()
        for (h, r, t) in new_ordered_train_data:
            value = 0
            if (h, t) in train_edges_betweenness_dict:
                value = train_edges_betweenness_dict[(h, t)]
            elif (t, h) in train_edges_betweenness_dict:
                value = train_edges_betweenness_dict[(t, h)]
            tmp_ordered_train_data.append((h, r, t, value))
        tmp_ordered_train_data.sort(key=lambda x: x[3], reverse=True) # 按从大到小排序
        new_ordered_train_data = list(map(lambda x: (x[0], x[1], x[2]), tmp_ordered_train_data))
        assert len(new_ordered_train_data) == ori_len
        return new_ordered_train_data

    def ordered_by_nodes_degree(self, new_ordered_train_data, ss_id):
        """ Intra-hierarchical: Sort by degree centrality from highest to lowest """
        ori_len = len(new_ordered_train_data)
        train_nodes_degree_path = self.args.data_path + str(ss_id) + "/" + "train_nodes_degree.txt"
        nodes = dict()
        with open(train_nodes_degree_path, "r", encoding="utf-8") as rf:
            for line in rf.readlines():
                line = line.strip()
                line_list = line.split("\t")
                node, value = int(line_list[0]), float(line_list[1])
                nodes[node] = value
        tmp_ordered_train_data = list()
        for (h, r, t) in new_ordered_train_data:
            v = max(nodes[h], nodes[t])
            tmp_ordered_train_data.append((h, r, t, v))
        tmp_ordered_train_data.sort(key=lambda x: x[3], reverse=True)
        new_ordered_train_data = list(map(lambda x: (x[0], x[1], x[2]), tmp_ordered_train_data))
        assert ori_len == len(new_ordered_train_data)
        return new_ordered_train_data

    def ordered_by_nodes_degree_and_edges(self, new_ordered_train_data, ss_id):
        """ Intra-hierarchical: Sorts by the centrality of nodes and the mesonumber of edges """
        ori_len = len(new_ordered_train_data)
        train_edges_betweenness_path = self.args.data_path + str(ss_id) + "/" + "train_edges_betweenness.txt"
        train_edges_betweenness_dict = dict()
        with open(train_edges_betweenness_path, "r", encoding="utf-8") as rf:
            for line in rf.readlines():
                line_list = line.strip().split("\t")
                node1, node2, value = int(line_list[0]), int(line_list[1]), float(line_list[2])
                train_edges_betweenness_dict[(node1, node2)] = value
                train_edges_betweenness_dict[(node2, node1)] = value
        nodes = dict()
        train_nodes_degree_path = self.args.data_path + str(ss_id) + "/" + "train_nodes_degree.txt"
        with open(train_nodes_degree_path, "r", encoding="utf-8") as rf:
            for line in rf.readlines():
                line = line.strip()
                line_list = line.split("\t")
                node, value = int(line_list[0]), float(line_list[1])
                nodes[node] = value
        tmp_ordered_train_data = list()
        for (h, r, t) in new_ordered_train_data:
            v = max(nodes[h], nodes[t])
            if (h, t) in train_edges_betweenness_dict:
                v += train_edges_betweenness_dict[(h, t)]
            elif (t, h) in train_edges_betweenness_dict:
                v += train_edges_betweenness_dict[(t, h)]
            tmp_ordered_train_data.append((h, r, t, v))
        tmp_ordered_train_data.sort(key=lambda x: x[3], reverse=True)
        new_ordered_train_data = list(map(lambda x: (x[0], x[1], x[2]), tmp_ordered_train_data))
        assert ori_len == len(new_ordered_train_data)
        return new_ordered_train_data

    def generate_layers(self, args):
        """ 1. read datasets """
        hr2t_all = {}
        train_all, valid_all, test_all = list(), list(), list()
        for ss_id in range(int(self.args.snapshot_num)):
            self.new_entities = set()
            train_facts = load_fact(self.args.data_path + str(ss_id) + "/" + "train.txt")
            valid_facts = load_fact(self.args.data_path + str(ss_id) + "/" + "valid.txt")
            test_facts = load_fact(self.args.data_path + str(ss_id) + "/" + "test.txt")
            self.expend_entity_relation(train_facts)
            self.expend_entity_relation(valid_facts)
            self.expend_entity_relation(test_facts)
            train = self.fact2id(train_facts)
            valid = self.fact2id(valid_facts, order=True)
            test = self.fact2id(test_facts, order=True)
            edge_h, edge_r, edge_t = [], [], []
            edge_h, edge_r, edge_t = self.expand_kg(train, 'train', edge_h, edge_r, edge_t, hr2t_all)
            edge_h, edge_r, edge_t = self.expand_kg(valid, 'valid', edge_h, edge_r, edge_t, hr2t_all)
            edge_h, edge_r, edge_t = self.expand_kg(test, 'test', edge_h, edge_r, edge_t, hr2t_all)
            train_all += train
            valid_all += valid
            test_all += test
            self.store_snapshot(ss_id, train, train_all, valid, valid_all, test, test_all, edge_h, edge_r, edge_t, hr2t_all)
            self.new_entities.clear()
        """ 2. Sort the first snapshot by betweenness of edge """
        train_data = deepcopy(self.snapshots[0].train)
        ori_len = len(train_data)
        train_edges_betweenness_path = self.args.data_path + str(0) + "/" + "train_edges_betweenness.txt"
        train_edges_betweenness_dict = dict()
        with open(train_edges_betweenness_path, "r", encoding="utf-8") as rf:
            for line in rf.readlines():
                line_list = line.strip().split("\t")
                node1, node2, value = int(line_list[0]), int(line_list[1]), float(line_list[2])
                train_edges_betweenness_dict[(node1, node2)] = value
                train_edges_betweenness_dict[(node2, node1)] = value
        tmp_train_data = list()
        for (h, r, t) in train_data:
            value = 0
            if (h, t) in train_edges_betweenness_dict:
                value = train_edges_betweenness_dict[(h, t)]
            elif (t, h) in train_edges_betweenness_dict:
                value = train_edges_betweenness_dict[(t, h)]
            tmp_train_data.append((h, r, t, value))
        tmp_train_data.sort(key=lambda x: x[3], reverse=True)
        train_data = list(map(lambda x: (x[0], x[1], x[2]), tmp_train_data))
        assert len(train_data) == ori_len
        train_data_path = self.args.data_path + str(0) + "/" + self.args.multi_layers_path
        with open(train_data_path, "w", encoding="utf-8") as wf:
            for (h, r, t) in train_data:
                wf.write(self.id2entity[h])
                wf.write("\t")
                wf.write(self.id2relation[r])
                wf.write("\t")
                wf.write(self.id2entity[t])
                wf.write("\n")
        """ 3. Sort other snapshots and write in files """
        for ss_id in range(1, int(self.args.snapshot_num)):
            train_data = deepcopy(self.snapshots[ss_id].train)
            train_data_len = len(train_data)
            last_entity_num = self.snapshots[ss_id - 1].num_ent
            old_entities = set([i for i in range(last_entity_num)])
            ordered_train_data = []
            flag = True
            lay_id = 1
            while flag:
                """ Step1: add triples with at leat one old entity """
                new_entities_ = set()
                new_ordered_train_data = list()
                for (h, r, t) in train_data:
                    if h in old_entities or t in old_entities:
                        new_ordered_train_data.append((h, r, t))
                        if h not in old_entities:
                            new_entities_.add(h)
                        if t not in old_entities:
                            new_entities_.add(t)
                if len(new_ordered_train_data) == 0:
                    break

                new_ordered_train_data = self.ordered_by_nodes_degree_and_edges(new_ordered_train_data, ss_id)
                """ Step2: update ordered_train_data and train_data """
                ordered_train_data += new_ordered_train_data
                train_data = list(filter(lambda x: x[0] not in old_entities and x[2] not in old_entities, train_data))
                """ Step3: update old graphs """
                old_entities = (old_entities | new_entities_)

                lay_id += 1
            """ Step4: add isolated subgraphs """

            if len(train_data):
                ordered_train_data += train_data
            assert train_data_len == len(ordered_train_data)
            """ Step5: write in files """
            ordered_data_path = self.args.data_path + str(ss_id) + "/" + self.args.multi_layers_path
            with open(ordered_data_path, "w", encoding="utf-8") as wf:
                for (h, r, t) in ordered_train_data:
                    wf.write(self.id2entity[h])
                    wf.write("\t")
                    wf.write(self.id2relation[r])
                    wf.write("\t")
                    wf.write(self.id2entity[t])
                    wf.write("\n")
        """ 4. reset datasets setting"""
        self.args = args
        self.num_ent, self.num_rel = 0, 0
        self.entity2id, self.id2entity, self.relation2id, self.id2relation = {}, {}, {}, {}
        self.relationid2invid = {}
        self.snapshots = {i: SnapshotIncDE(self.args) for i in range(int(self.args.snapshot_num))}

    def load_data(self):
        """ Load data from all snapshots """
        hr2t_all = {}
        train_all, valid_all, test_all = [], [], []
        for ss_id in range(int(self.args.snapshot_num)):
            self.new_entities = set() # all entities in this snapshot
            """ Step 1: (h, r, t) """
            if self.args.use_multi_layers and ss_id:
                try:
                    train_facts = load_fact(self.args.data_path + str(ss_id) + "/" + self.args.multi_layers_path)
                except:
                    train_facts = load_fact(self.args.data_path + str(ss_id) + "/" + "train.txt")
                train_facts = load_fact(self.args.data_path + str(ss_id) + "/" + "train.txt")
                print("Using multi layers data")
            else:
                train_facts = load_fact(self.args.data_path + str(ss_id) + "/" + "train.txt")
            valid_facts = load_fact(self.args.data_path + str(ss_id) + "/" + "valid.txt") # valid -> test
            test_facts = load_fact(self.args.data_path + str(ss_id) + "/" + "test.txt")

            """ Step 2: h -> h_id, r -> r_id, t -> t_id """
            self.expend_entity_relation(train_facts)
            self.expend_entity_relation(valid_facts)
            self.expend_entity_relation(test_facts)

            """ Step 3: (h, r, t) -> (h_id, r_id, t_id) """
            train = self.fact2id(train_facts)
            valid = self.fact2id(valid_facts, order=True)
            test = self.fact2id(test_facts, order=True)
            
            """ Step 4: [h1, h2, ..., hn] (train set),
                        [r1, r2, ..., rn] (train set),
                        [t1, t2, ..., tn] (train set),
                        {(h1, r1): t1, (h2, r2): t2, ..., (hn, rn): tn}
            """
            edge_h, edge_r, edge_t = [], [], []
            edge_h, edge_r, edge_t = self.expand_kg(train, 'train', edge_h, edge_r, edge_t, hr2t_all)
            edge_h, edge_r, edge_t = self.expand_kg(valid, 'valid', edge_h, edge_r, edge_t, hr2t_all)
            edge_h, edge_r, edge_t = self.expand_kg(test, 'test', edge_h, edge_r, edge_t, hr2t_all)

            """ Step 5: Get all (h_id, r_id, t_id) """
            train_all += train
            valid_all += valid
            test_all += test

            """ Step 6: Store this snapshot """
            self.store_snapshot(ss_id, train, train_all, valid, valid_all, test, test_all, edge_h, edge_r, edge_t, hr2t_all)
            self.new_entities.clear()

    def store_snapshot(self, ss_id, train, train_all, valid, valid_all, test, test_all, edge_h, edge_r, edge_t, hr2t_all):
        """ Store num_ent, num_rel """
        self.snapshots[ss_id].num_ent = deepcopy(self.num_ent)
        self.snapshots[ss_id].num_rel = deepcopy(self.num_rel)

        """ Store (h, r, t) """
        self.snapshots[ss_id].train = deepcopy(train)
        self.snapshots[ss_id].train_all = deepcopy(train_all)
        self.snapshots[ss_id].valid = deepcopy(valid)
        self.snapshots[ss_id].valid_all = deepcopy(valid_all)
        self.snapshots[ss_id].test = deepcopy(test)
        self.snapshots[ss_id].test_all = deepcopy(test_all)

        """ Store [h1, h2, ..., hn], [r1, r2, ..., rn], [t1, t2, ..., tn] """

        """ Store some special things """
        self.snapshots[ss_id].hr2t_all = deepcopy(hr2t_all)


    def expand_kg(self, facts, split, edge_h, edge_r, edge_t, hr2t_all):
        """ Get edge_index and edge_type for GCN and hr2t_all for filter golden facts """
        def add_key2val(dict, key, val):
            """ add {key: val} to dict"""
            if key not in dict.keys():
                dict[key] = set()
            dict[key].add(val)

        for (h, r, t) in facts:
            self.new_entities.add(h)
            self.new_entities.add(t)
            if split == "train":
                """ edge_index """
                edge_h.append(h)
                edge_r.append(r)
                edge_t.append(t)
            """ hr2t """
            add_key2val(hr2t_all, (h, r), t)
            add_key2val(hr2t_all, (t, self.relationid2invid[r]), h)
        return edge_h, edge_r, edge_t

    def fact2id(self, facts, order=False):
        """ (h, r, t) -> (h_id, r_id, t_id) """
        fact_id = []
        if order:
            i = 0
            while len(fact_id) < len(facts):
                for (h, r, t) in facts:
                    if self.relation2id[r] == i:
                        fact_id.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))
                i += 2
        else:
            for (h, r, t) in facts:
                fact_id.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))
        return fact_id

    def expend_entity_relation(self, facts):
        """ extract entities and relations from new facts """
        for (h, r, t) in facts:
            """ extract entities """
            if h not in self.entity2id.keys():
                self.entity2id[h] = self.num_ent
                if self.args.use_multi_layers:
                    self.id2entity[self.num_ent] = h
                self.num_ent += 1
            if t not in self.entity2id.keys():
                self.entity2id[t] = self.num_ent
                if self.args.use_multi_layers:
                    self.id2entity[self.num_ent] = t
                self.num_ent += 1

            """ extract relations """
            if r not in self.relation2id.keys():
                self.relation2id[r] = self.num_rel
                if self.args.use_multi_layers:
                    self.id2relation[self.num_rel] = r
                self.relation2id[r + "_inv"] = self.num_rel + 1
                if self.args.use_multi_layers:
                    self.id2relation[self.num_rel + 1] = r + "_inv"
                self.relationid2invid[self.num_rel] = self.num_rel + 1
                self.relationid2invid[self.num_rel + 1] = self.num_rel
                self.num_rel += 2


class SnapshotIncDE():
    def __init__(self, args) -> None:
        self.args = args
        self.num_ent, self.num_rel = 0, 0
        self.train, self.train_all, self.valid, self.valid_all, self.test, self.test_all = [], [], [], [], [], []
        self.edge_h, self.edge_r, self.edge_t = [], [], []
        self.hr2t_all = {}
        self.edge_index, self.edge_type = None, None
        self.new_entities = []