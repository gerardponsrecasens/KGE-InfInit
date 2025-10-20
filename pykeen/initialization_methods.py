import torch
import pickle


# Function to read triples from the file and return as a list of lists
def read_triples(file_path):
    triples = []
    with open(file_path, 'r') as file:
        for line in file:
            head, relation, tail = line.strip().split('\t')
            triples.append([head, relation, tail]) 
    return triples

# Function to retrieve all triples that contain the given entity (either as head or tail)
def find_triples_by_entity(triples, entity):
    matching_triples = []
    for head, relation, tail in triples:
        if head == entity or tail == entity:
            matching_triples.append([head, relation, tail])
    return matching_triples



def init_class_avg(old_ent_to_id,old_ent_emb,new_ent_to_id,new_ent_emb,emb_dimension,pruned_dict,random_noise, std_frac):

    '''
    Function to create an initialization that moves each entity to the average embedding of the classes it belongs to.
    The embeddin of the classes it belongs to is computed by averaging all the entities that belong to the class.

    :param old_ent_to_id: Entity to ID dictionary for the previous snapshot
    :param old_ent_emb: Embedddings obtained in the previous snapshot
    :param new_ent_to_id: Entity to ID dictionary for the current snapshot
    :param new_ent_emb: Embedddings for the current snapshot
    :param emb_dimension: Embedding dimension
    :param pruned dict: If we want to use only the classes that are leaves
    :param random_noise: If we want to generate random noise around the placed new embedding
    :param std_frac: How many std we want to add in the random noise
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_path = 'dictionary_db.pkl'

    with open(file_path, 'rb') as file:
        class_dict = pickle.load(file) # {'ent_name':[type1,type2],...}

    class_to_entities = {} # {'type1':[0,9,245],..} of entities already in the KG

    for entity, idx in old_ent_to_id.items():
        classes = class_dict[entity]

        for c in classes:
            if c in class_to_entities.keys():
                class_to_entities[c].append(idx)
            else:
                class_to_entities[c] = [idx]


    # Assign to each class its average embedding
    class_avg = {} # {'type1': [0.1,0.32,...,0.9],...}

    for c, idx_list in class_to_entities.items():
        initial = torch.zeros([1,emb_dimension]).to(device)

        for idx in idx_list:
            initial = initial + old_ent_emb[idx]

        class_avg[c] = initial/len(idx_list)


    class_std = {}
    for c, idx_list in class_to_entities.items():
        total = torch.zeros([1,emb_dimension]).to(device)

        for idx in idx_list:
            total = total + torch.abs(old_ent_emb[idx]-class_avg[c])**2

        class_std[c] = (total/len(idx_list))**0.5

    with torch.no_grad():
        for ent,idx in new_ent_to_id.items():
            if ent not in old_ent_to_id.keys():
                ent_classes = class_dict[ent]

                previous_classes = [i for i in ent_classes if i in class_to_entities.keys()]

                if len(previous_classes) != 0: #some entities do not have any class assigned
                    initial = torch.zeros([1,emb_dimension]).to(device)
                    initial_std = torch.zeros([1,emb_dimension]).to(device)

                    for ent_class in previous_classes:
                        initial = initial + class_avg[ent_class]
                        initial_std = initial_std + class_std[ent_class]
                    
                    initial = initial/len(previous_classes)
                    initial_std = initial_std/len(previous_classes)

                    if random_noise:
                        initial = initial + std_frac*initial_std*torch.randn(1,emb_dimension).to(device)

                    new_ent_emb[idx] = initial

    return new_ent_emb
    


def init_on_triples(s,old_ent_to_id,old_ent_emb,old_rel_to_id, old_rel_emb,new_ent_to_id,new_ent_emb,emb_dimension,path,random_noise,std_frac):
    '''
    Function to initialize triples based on the model prediction of the triples it participates in

    :param s: Current snapshot
    :param old_ent_to_id: Entity to ID dictionary for the previous snapshot
    :param old_ent_emb: Embedddings obtained in the previous snapshot for entities
    :param old_rel_to_id: Relation to ID dictionary for the previous snapshot
    :param old_rel_emb: Embedddings obtained in the previous snapshot for relations
    :param new_ent_to_id: Entity to ID dictionary for the current snapshot
    :param new_ent_emb: Embedddings for the current snapshot
    :param emb_dimension: Embedding dimension

    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_path = path+str(s)+'/train.txt'
    triples = read_triples(file_path)

    with torch.no_grad():
        for ent,idx in new_ent_to_id.items():
            if ent not in old_ent_to_id.keys(): 
                matching_triples = find_triples_by_entity(triples, ent)

                ct = 0
                initial = torch.zeros([1,emb_dimension]).to(device)

                for triple in matching_triples:

                    head, relation, tail = triple

                    if head == ent:
                        if tail in old_ent_to_id.keys() and relation in old_rel_to_id.keys(): #They previosuly exist
                            ct +=1
                            r_idx = old_rel_to_id[relation]
                            t_idx = old_ent_to_id[tail]

                            initial += -old_rel_emb[r_idx]+old_ent_emb[t_idx]
                    else:
                        if head in old_ent_to_id.keys() and relation in old_rel_to_id.keys(): #They previosuly exist
                            ct +=1
                            r_idx = old_rel_to_id[relation]
                            h_idx = old_ent_to_id[head]
                            initial += old_rel_emb[r_idx]+old_ent_emb[h_idx]
                
                if ct !=0:
                    initial = initial/ct
                    if random_noise:
                        initial += std_frac*torch.randn(1,emb_dimension).to(device)
                    new_ent_emb[idx] = initial

    return new_ent_emb


def init_class_avg_RotatE(old_ent_to_id,old_ent_emb,new_ent_to_id,new_ent_emb,emb_dimension,pruned_dict,random_noise, std_frac):

    '''
    Function to create an initialization that moves each entity to the average embedding of the classes it belongs to.
    The embeddin of the classes it belongs to is computed by averaging all the entities that belong to the class.

    :param old_ent_to_id: Entity to ID dictionary for the previous snapshot
    :param old_ent_emb: Embedddings obtained in the previous snapshot
    :param new_ent_to_id: Entity to ID dictionary for the current snapshot
    :param new_ent_emb: Embedddings for the current snapshot
    :param emb_dimension: Embedding dimension
    :param pruned dict: If we want to use only the classes that are leaves
    :param random_noise: If we want to generate random noise around the placed new embedding
    :param std_frac: How many std we want to add in the random noise
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the DBpedia class dictionaries
    if pruned_dict:
        file_path = 'dictionary_db_pruned.pkl'
    else:
        file_path = 'dictionary_db.pkl'

    with open(file_path, 'rb') as file:
        class_dict = pickle.load(file) # {'ent_name':[type1,type2],...}

    class_to_entities = {} # {'type1':[0,9,245],..} of entities already in the KG

    for entity, idx in old_ent_to_id.items():
        classes = class_dict[entity]

        for c in classes:
            if c in class_to_entities.keys():
                class_to_entities[c].append(idx)
            else:
                class_to_entities[c] = [idx]


    # Assign to each class its average embedding
    class_avg = {} # {'type1': [0.1,0.32,...,0.9],...}

    for c, idx_list in class_to_entities.items():
        initial = torch.zeros([1,emb_dimension], dtype=torch.complex64).to(device)

        for idx in idx_list:
            initial += old_ent_emb[idx]

        class_avg[c] = initial/len(idx_list)


    class_std = {}
    for c, idx_list in class_to_entities.items():
        total = torch.zeros([1,emb_dimension],dtype=torch.complex64).to(device)

        for idx in idx_list:
            total += torch.abs(old_ent_emb[idx]-class_avg[c])**2

        class_std[c] = (total/len(idx_list))**0.5

    with torch.no_grad():
        for ent,idx in new_ent_to_id.items():
            if ent not in old_ent_to_id.keys():
                ent_classes = class_dict[ent]

                previous_classes = [i for i in ent_classes if i in class_to_entities.keys()]

                if len(previous_classes) != 0: #some entities do not have any class assigned
                    initial = torch.zeros([1,emb_dimension],dtype=torch.complex64).to(device)
                    initial_std = torch.zeros([1,emb_dimension],dtype=torch.complex64).to(device)

                    for ent_class in previous_classes:
                        initial += class_avg[ent_class]
                        initial_std += class_std[ent_class]
                    
                    initial = initial/len(previous_classes)
                    initial_std = initial_std/len(previous_classes)

                    if random_noise:
                        initial += std_frac*initial_std*torch.randn(1,emb_dimension).to(device)

                    new_ent_emb[idx] = initial

    return new_ent_emb