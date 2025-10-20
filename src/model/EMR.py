from .BaseModel import *
from .initialization import *
import pickle

class EMR(BaseModel):
    def __init__(self, args, kg):
        super(EMR, self).__init__(args, kg)
        self.ce = nn.CrossEntropyLoss()
        self.args.n_memories = 1000

    def pre_snapshot(self):
        '''
        Prepare for training on this snapshot
        '''

        if self.args.snapshot == 0:
            '''sample old facts'''
            self.initialize_memory()
        else:
            '''update old facts'''
            self.update_memory()

    def initialize_memory(self):
        '''sample old facts in first training set'''
        train_data = self.kg.snapshots[0].train_new
        self.memory_data = random.sample(train_data, self.args.n_memories)

    def update_memory(self):
        '''update a half of old facts'''
        random.shuffle(self.memory_data)
        train_data = self.kg.snapshots[self.args.snapshot].train_new
        self.memory_data = self.memory_data[:self.args.n_memories//2] + random.sample(train_data, self.args.n_memories//2)

    def switch_snapshot(self):
        '''prepare for next snapshot'''
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = torch.nn.Parameter(self.ent_embeddings.weight.data)
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = torch.nn.Parameter(self.rel_embeddings.weight.data)
        
        
        init = self.args.init

        if init != 0:

            sd_frac = self.args.RN

            # New entities added in the snapshot
            with open('./dicts/'+self.args.dataset+'_new_entities.pkl', 'rb') as file:
                new_entities = pickle.load(file)
            new_entities_snapshot = new_entities[self.args.snapshot+1]

            # Entities for which embeddings are already generated
            old_entities = []
            for previous_snapshot in range(self.args.snapshot+1):
                old_entities += new_entities[previous_snapshot]

            with open('./dicts/'+self.args.dataset+'_new_relations.pkl', 'rb') as file:
                new_relations = pickle.load(file)
            new_relations_snapshot = new_relations[self.args.snapshot+1] #We use +1 as this is before going to new snapshot

            # Load the old entities
            old_relations = []
            for previous_snapshot in range(self.args.snapshot+1):
                old_relations += new_relations[previous_snapshot]
            

            if init == 1:
                new_ent_embeddings = schema_initialization_ent(self.args, self.kg, new_ent_embeddings, old_entities, new_entities_snapshot, sd_frac)
                new_rel_embeddings = schema_initialization_rel(self.args, self.kg, new_rel_embeddings, old_relations, new_relations_snapshot, sd_frac) 
            if init == 3:
                new_ent_embeddings = model_initialization_ent(self.args, self.kg, new_ent_embeddings, new_rel_embeddings, old_entities, new_entities_snapshot, sd_frac)
                new_rel_embeddings = model_initialization_rel(self.args, self.kg, new_ent_embeddings, new_rel_embeddings, old_entities, new_relations_snapshot, sd_frac)
                    
        self.ent_embeddings.weight = torch.nn.Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = torch.nn.Parameter(new_rel_embeddings)

    def replay(self, x, label):
        '''replay old facts'''
        pt_triples, pt_label = self.corrupt(self.memory_data)
        pt_triples, pt_label = torch.LongTensor(pt_triples).to(self.args.device), torch.Tensor(pt_label).to(self.args.device)
        '''merge old and new facts'''
        x = torch.cat([x, pt_triples], dim=0)
        '''get loss'''
        label = torch.cat([label, pt_label], dim=0)
        head, rel, tail = x[:, 0], x[:, 1], x[:, 2]
        loss = self.new_loss(head, rel, tail, label)
        return loss

    def corrupt(self, facts):
        '''
        Create negative samples by randomly corrupt subject or object entity
        :param triples:
        :return: negative samples
        '''
        ss_id = self.args.snapshot
        label = []
        facts_ = []
        for fact in facts:
            s, r, o = fact[0], fact[1], fact[2]
            prob = 0.5
            neg_s = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
            neg_o = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
            pos_s = np.ones_like(neg_s) * s
            pos_o = np.ones_like(neg_o) * o
            rand_prob = np.random.rand(self.args.neg_ratio)
            sub = np.where(rand_prob > prob, pos_s, neg_s)
            obj = np.where(rand_prob > prob, neg_o, pos_o)
            facts_.append((s, r, o))
            label.append(1)
            for ns, no in zip(sub, obj):
                facts_.append((ns, r, no))
                label.append(-1)
        return facts_, label


class TransE(EMR):
    def __init__(self, args, kg):
        super(TransE, self).__init__(args, kg)

    def loss(self, head, rel, tail=None, label=None):
        '''
        :param head: subject entity
        :param rel: relation
        :param tail: object entity
        :param label: positive or negative facts
        :return: new facts loss
        '''
        x = torch.cat([head.unsqueeze(1), rel.unsqueeze(1), tail.unsqueeze(1)], dim=1)
        if self.args.snapshot > 0:
            new_loss = self.replay(x, label)
        else:
            new_loss = self.new_loss(head, rel, tail, label)
        return new_loss




