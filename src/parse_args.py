import argparse
parser = argparse.ArgumentParser(description='Parser For Arguments',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# training control
parser.add_argument('-snapshot_num', dest='snapshot_num', default=5, help='The snapshot number of the dataset')
parser.add_argument('-dataset', dest='dataset', default='HYBRID', help='dataset name')
parser.add_argument('-gpu', dest='gpu', default=0)
parser.add_argument('-loss_name', dest='loss_name', default='Margin', help='Margin: pairwise margin loss')
parser.add_argument('-train_new', dest='train_new', default=True, help='True: Training on new facts; False: Training on all seen facts')
parser.add_argument('-skip_previous', dest='skip_previous', default='False', help='Allow re-training and snapshot_only models skip previous training')

# model setting
parser.add_argument('-lifelong_name', dest='lifelong_name', default='LKGE')
parser.add_argument('-optimizer_name', dest='optimizer_name', default='Adam')
parser.add_argument('-embedding_model', dest='embedding_model', default='TransE')
parser.add_argument('-epoch_num', dest='epoch_num', default=200, help='max epoch num')
parser.add_argument('-incremental_epochs', dest='incremental_epochs', default=200, help='max epoch num in incremental') #200
parser.add_argument('-margin', dest='margin', default=8.0, help='The margin of MarginLoss')
parser.add_argument('-batch_size', dest='batch_size', default=2048, help='Mini-batch size')
parser.add_argument('-learning_rate', dest='learning_rate', default=0.0001)
parser.add_argument('-emb_dim', dest='emb_dim', default=200, help='embedding dimension') #200
parser.add_argument('-l2', dest='l2', default=0.0, help='optimizer l2')
parser.add_argument('-neg_ratio', dest='neg_ratio', default=10, help='the ratio of negative/positive facts')
parser.add_argument('-patience', dest='patience', default=3, help='early stop step')
parser.add_argument('-regular_weight', dest='regular_weight', default=0.01, help='Regularization strength: alpha')
parser.add_argument('-reconstruct_weight', dest='reconstruct_weight', default=0.1, help='The weight of MAE loss: beta')

# Initialization
parser.add_argument('-init', dest='init',type=int, default=0)
parser.add_argument('-RN', dest='RN',type=float, default=0, help = 'Random Noise for the Initialization')
parser.add_argument('-reuse_0', dest='reuse_0',action='store_true', help='To load and reuse the model on first snapshot')


#LKGE
parser.add_argument('-using_regular_loss', dest='using_regular_loss', default='True')
parser.add_argument('-using_reconstruct_loss', dest='using_reconstruct_loss', default='True')
parser.add_argument('-using_embedding_transfer', dest='using_embedding_transfer', default='False')
parser.add_argument('-using_finetune', dest='using_finetune', default='True')

#incDE
parser.add_argument("-using_embedding_distill", dest="using_embedding_distill", default=True, help="Using Embedding distill or not") # for use
parser.add_argument("-embedding_distill_weight", dest="embedding_distill_weight", default=0.1, help="weight of distllation")
parser.add_argument("-muti_embedding_distill_weight", dest="muti_embedding_distill_weight", default=1, help="weight of multi distllation")
parser.add_argument("-using_multi_embedding_distill", dest="using_multi_embedding_distill", default=False, help="use multi distillation or not")
parser.add_argument("-multi_distill_num", dest="multi_distill_num", default=3, help="num of distills")
parser.add_argument("-using_structure_distill", dest="using_structure_distill", default=False, help="Using structure distill or not")
parser.add_argument("-structure_distill_weight", dest="structure_distill_weight", default=0.1, help="The weight of structure weight")
parser.add_argument("-using_score_distill", dest="using_score_distill", default=False, help="Using score distill or not")
parser.add_argument("-score_distill_weight", dest="score_distill_weight", default=1, help="The weight of score distill")
parser.add_argument("-num_old_triples", dest="num_old_triples", default=20000, help="Num of old triples")
parser.add_argument("-using_reply", dest="using_reply", default=False, help="Use reply or not")
parser.add_argument("-reply_loss_weight", dest="reply_loss_weight", default=0.1, help="The weight of reply loss")
parser.add_argument("-using_contrast_distill", dest="using_contrast_distill", default=False, help="Using contrast distill or not")
parser.add_argument("-contrast_loss_weight", dest="contrast_loss_weight", default=0.1, help="The weight of contrast loss")
parser.add_argument("-use_multi_layers", dest="use_multi_layers", default=True, help="Use multi layers or not") # Multi-layer distillation
parser.add_argument("-multi_layers_path", dest="multi_layers_path", default="train_sorted_by_edges_betweenness.txt", help="New_path")
parser.add_argument("-multi_layer_weight", dest="multi_layer_weight", default=0.01, help="The weight of multi layer weight")
parser.add_argument("-use_two_stage", dest="use_two_stage", default=True, help="Use two stage distill or not") # two-stage
parser.add_argument("-two_stage_epoch_num", dest="two_stage_epoch_num", default=20, help="Num of two stage epoch")
parser.add_argument("-using_all_data", dest="using_all_data", default=False, help="Using all data or not")
parser.add_argument("-using_relation_distill", dest="using_relation_distill", default=False, help="Using relation distill or not")
parser.add_argument("-using_mask_weight", dest="using_mask_weight", default=True, help="Using mask weight or not")  # dynamic weights
parser.add_argument("-using_different_weights", dest="using_different_weights", default=True, help="Using different weights or not")
parser.add_argument("-using_test", dest="using_test", default=False, help="test mode") # for debug
parser.add_argument("-first_training", dest="first_training", default=True, help="First training on a dataset") # first training on a dataset
parser.add_argument("-without_hier_distill", dest="without_hier_distill", default=False, help="Without hier distillation") # ablation
parser.add_argument("-without_two_stage", dest="without_two_stage", default=False, help="Without two stage") # ablation
parser.add_argument("-without_multi_layers", dest="without_multi_layers", default=False, help="Without multi layers") # ablation
parser.add_argument("-record", dest="record", default=False, help="Record the loss of different layers") # explore
parser.add_argument("-predict_result", dest="predict_result", default=False, help="The result of predict") # explore

# others
parser.add_argument('-save_path', dest='save_path', default='./checkpoint/')
parser.add_argument('-data_path', dest='data_path', default='./data/')
parser.add_argument('-log_path', dest='log_path', default='./logs/')
parser.add_argument('-num_layer', dest='num_layer', default=1, help='MAE layer')
parser.add_argument('-num_workers', dest='num_workers', default=1)
parser.add_argument('-valid_metrics', dest='valid_metrics', default='mrr')
parser.add_argument('-valid', dest='valid', default=True, help='indicator of test or valid')
parser.add_argument('-note', dest='note', default='', help='The note of log file name')
parser.add_argument('-seed', dest='seed', default=55)
parser.add_argument("-random_seed", dest="random_seed", default=3407)
args = parser.parse_args()