from .BaseModel import *
from .initialization import *
import pickle


class finetune(BaseModel):
    def __init__(self, args, kg):
        super(finetune, self).__init__(args, kg)

    def switch_snapshot(self):
        '''expand embeddings for new entities and relations '''
        ent_embeddings, rel_embeddings = self.expand_embedding_size()

        '''inherit learned parameters'''
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = torch.nn.Parameter(
            self.ent_embeddings.weight.data)
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = torch.nn.Parameter(
            self.rel_embeddings.weight.data)
        
       
        
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



class TransE(finetune):
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
        new_loss = self.new_loss(head, rel, tail, label)
        return new_loss










