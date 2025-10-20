from .BaseModel import *
from .initialization import *
import pickle


class EWC(BaseModel):
    '''
    We refer to the implementation of https://github.com/joansj/hat/blob/master/src/approaches/ewc.py.
    '''
    def __init__(self, args, kg):
        super(EWC, self).__init__(args, kg)
        self.mse_loss = nn.MSELoss(size_average=False)

        # initialize fisher matrix
        self.fisher = dict()
        self.fisher_old = dict()
        for n, p in self.named_parameters():
            n = n.replace('.','_')
            self.fisher[n] = 0 * p.data

    def pre_snapshot(self):
        if self.args.snapshot >= 0:
            self.fisher_old = {}
            for n, _ in self.named_parameters():
                n = n.replace('.','_')
                self.fisher_old[n] = self.fisher[n].clone()

    def switch_snapshot(self):
        '''
        Prepare for the training of next snapshot
        EWC:
            1. expand the fisher matrix for new entity and relation embeddings;
            2. store learn parameters;
            3. inherit the model learned from previous snapshot and prepare embeddings for new entities and relations.
            4. store old fisher matrix
        :return:
        '''

        '''1. expand fisher matrix for new entity and relation embeddings'''
        for n, _ in self.named_parameters():
            n = n.replace('.', '_')
            self.fisher[n] /= len(self.kg.snapshots[self.args.snapshot].train_new)
            self.fisher[n] = torch.cat([(self.fisher[n][:self.fisher_old[n].size(0)] + self.fisher_old[n] * self.args.snapshot+1) / (self.args.snapshot + 2),self.fisher[n][self.fisher_old[n].size(0):]],dim=0)

        '''2. store learned parameters'''
        self.store_old_parameters()

        '''3. inherit the model learned from previous snapshot and prepare embeddings for new entities and relations'''
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

        '''4. store old fisher matrix '''
        if self.args.snapshot >= 0:
            self.fisher_old = {}
            for n, _ in self.named_parameters():
                n = n.replace('.','_')
                self.fisher_old[n] = self.fisher[n].clone()

    def ewc_loss(self):
        '''
        Get regularization loss for all old paramters to constraint the update of old paramters.
        '''
        losses = []
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            new_data = param
            old_data = getattr(self, 'old_data_{}'.format(name))
            if len(param.shape) == 2:
                losses.append(torch.sum((self.fisher[name][:old_data.size(0)])*(new_data[:old_data.size(0)]-old_data).pow(2)))
            else:
                losses.append(self.fisher[name]*self.mse_loss(new_data, old_data).sum())
        loss_reg = sum(losses)
        return loss_reg

    def epoch_post_processing(self, size=None):
        '''
        Process for next training iteration
        '''
        ''' update the fisher matrix for parameters '''
        for n, p in self.named_parameters():
            n = n.replace('.', '_')
            if p.grad is not None:
                if self.fisher[n].size(0) != p.grad.data.size(0):
                    self.fisher[n] = torch.cat([self.fisher[n], torch.zeros(p.grad.data.size(0)-self.fisher[n].size(0), self.fisher[n].size(1)).to(self.args.device)], dim=0)
                self.fisher[n] += size * p.grad.data.pow(2)


class TransE(EWC):
    def __init__(self, args, kg):
        super(TransE, self).__init__(args, kg)

    def loss(self, head, rel, tail=None, label=None):
        '''
        :param head: subject entity
        :param rel: relation
        :param tail: object entity
        :param label: positive or negative facts
        :return: new facts loss + regularization loss
        '''
        new_loss = self.new_loss(head, rel, tail, label)
        if self.args.snapshot > 0:
            ewc_loss = self.ewc_loss()
        else:
            ewc_loss = 0.0
        return new_loss + float(self.args.regular_weight) * ewc_loss




