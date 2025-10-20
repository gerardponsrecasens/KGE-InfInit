import torch
import pickle
from collections import defaultdict


def schema_initialization_ent(args, kg, new_ent_embeddings, old_entities, new_entities_snapshot, sd_frac):
    '''
    Implementation of the initialization proposed in Method Section of the paper.
    '''
    # Load the class dictionary
    with open('./dicts/dictionary_db.pkl', 'rb') as file:
        class_dict = pickle.load(file) #{'ent_name':[type1,type2],...}


    class_to_entities = defaultdict(list)
    for entity in old_entities:
        idx = kg.entity2id[entity]
        for c in class_dict[entity]:
            class_to_entities[c].append(idx)

    emb_dim = args.emb_dim
    device = args.device

    # Compute class averages and stds 
    class_avg = {}
    class_std = {}

    for c, idx_list in class_to_entities.items():
        idx_tensor = torch.tensor(idx_list, device=device)
        embeddings = new_ent_embeddings[idx_tensor]  # Shape: [N, emb_dim]

        avg = embeddings.mean(dim=0, keepdim=True)  # Shape: [1, emb_dim]
        std = embeddings.std(dim=0, unbiased=False, keepdim=True)  # Shape: [1, emb_dim]

        class_avg[c] = avg
        class_std[c] = std
    


    # Initialize new entity embeddings based on class averages
    for ent in new_entities_snapshot:
        idx = kg.entity2id[ent]
        ent_classes = class_dict[ent]

        # Only consider classes with prior entity embeddings
        prev_classes = [c for c in ent_classes if c in class_avg]

        if prev_classes:
            avg_stack = torch.cat([class_avg[c] for c in prev_classes], dim=0)  # [K, emb_dim]
            std_stack = torch.cat([class_std[c] for c in prev_classes], dim=0)  # [K, emb_dim]

            mean_avg = avg_stack.mean(dim=0, keepdim=True)
            mean_std = std_stack.mean(dim=0, keepdim=True)

            noise = torch.randn_like(mean_avg) * mean_std * sd_frac
            new_ent_embeddings[idx] = mean_avg + noise
    
    return new_ent_embeddings


def model_initialization_ent(args, kg, new_ent_embeddings, new_rel_embeddings, old_entities, new_entities_snapshot, sd_frac):

    with open('./dicts/'+args.dataset+'_new_relations.pkl', 'rb') as file:
        new_relations = pickle.load(file)

    # Load the old entities
    old_relations = []
    for previous_snapshot in range(args.snapshot+1):
        old_relations += new_relations[previous_snapshot]

    with open('./dicts/'+args.dataset+'_new_triples.pkl', 'rb') as file:
        new_triples = pickle.load(file)
    new_triples_snapshot = new_triples[args.snapshot+1]
    
    
    for ent in new_entities_snapshot:
        idx = kg.entity2id[ent]

        matching_triples = []
        for head, relation, tail in new_triples_snapshot:
            if head == ent or tail == ent:
                matching_triples.append([head, relation, tail])

        ct = 0
        initial = torch.zeros([1,args.emb_dim]).to(args.device).double()

        for triple in matching_triples:

            head, relation, tail = triple

            if head == ent:
                if tail in old_entities and relation in old_relations: #They previosuly exist
                    ct +=1
                    r_idx = kg.relation2id[relation]
                    t_idx = kg.entity2id[tail]

                    initial += -new_rel_embeddings[r_idx]+new_ent_embeddings[t_idx]
            else:
                if head in old_entities and relation in old_entities: #They previosuly exist
                    ct +=1
                    r_idx = kg.relation2id[relation]
                    h_idx = kg.entity2id[head]
                    initial += new_rel_embeddings[r_idx]+new_ent_embeddings[h_idx]
        
        if ct !=0:
            initial = initial/ct

            
            initial += sd_frac*torch.randn(1,args.emb_dim).to(args.device).double()

            new_ent_embeddings[idx] = initial
    return new_ent_embeddings




def schema_initialization_rel(args, kg, new_rel_embeddings, old_relations, new_relations_snapshot, sd_frac):
    '''
    Implementation of the initialization proposed in Method Section of the paper.
    '''
    # Load the class dictionary
    # Load the dataset that contains for each relation its classes
    with open('./dicts/rel_db.pkl', 'rb') as file:
        class_dict = pickle.load(file)
    # Create a dictionary that for each class has the relations (their idx) containing it
    class_to_relations = defaultdict(list)
    for relation in old_relations:
        idx = kg.relation2id[relation]
        for c in class_dict[relation]:
            class_to_relations[c].append(idx)
    
    emb_dim = args.emb_dim
    device = args.device

    # Compute class averages using efficient tensor operations
    class_avg = {}

    for c, idx_list in class_to_relations.items():
        idx_tensor = torch.tensor(idx_list, device=device)
        embeddings = new_rel_embeddings[idx_tensor]  # Shape: [N, emb_dim]

        avg = embeddings.mean(dim=0, keepdim=True)  # Shape: [1, emb_dim]

        class_avg[c] = avg

        # Initialize new relation embeddings based on class averages
        for rel in new_relations_snapshot:
            idx = kg.relation2id[rel] #idx of the relation
            rel_classes = class_dict[rel] #classes of the relation

            # Only consider classes with prior entity embeddings
            prev_classes = [c for c in rel_classes if c in class_avg]

            if prev_classes:
                avg_stack = torch.cat([class_avg[c] for c in prev_classes], dim=0)  # [K, emb_dim]

                mean_avg = avg_stack.mean(dim=0, keepdim=True)
                new_rel_embeddings[idx] = mean_avg
    
    return new_rel_embeddings


def model_initialization_rel(args, kg, new_ent_embeddings, new_rel_embeddings, old_entities, new_relations_snapshot, sd_frac):
    
    

    with open('./dicts/'+args.dataset+'_new_triples.pkl', 'rb') as file:
        new_triples = pickle.load(file)
    new_triples_snapshot = new_triples[args.snapshot+1]
    


    for rel in new_relations_snapshot:
        idx = kg.relation2id[rel]

        matching_triples = []
        for head, relation, tail in new_triples_snapshot:
            if relation == rel:
                matching_triples.append([head, relation, tail])

        ct = 0
        initial = torch.zeros([1,args.emb_dim]).to(args.device).double()

        for triple in matching_triples:

            head, relation, tail = triple

            
            if tail in old_entities and head in old_entities: #They previosuly exist
                ct +=1
                h_idx = kg.entity2id[head]
                t_idx = kg.entity2id[tail]

                initial += new_ent_embeddings[t_idx] -new_ent_embeddings[h_idx]
            
        
        if ct !=0:
            initial = initial/ct
            new_rel_embeddings[idx] = initial


    return new_rel_embeddings