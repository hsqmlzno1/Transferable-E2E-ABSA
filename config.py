
import argparse

parser = argparse.ArgumentParser(description="Transferable-E2E-TBSA")
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--selective', action='store_true')
parser.add_argument("-round", '-r', type=int, default=1, help="Round number")
parser.add_argument("-source_domain", '-s', type=str, choices=['laptop', 'rest', 'device', 'service'], default='laptop', help="source domain")
parser.add_argument("-target_domain", '-t', type=str, choices=['laptop', 'rest', 'device', 'service'], default='rest', help="target domain")
parser.add_argument("-model_name", '-m', type=str, default='AD-SAL', help="model name")

# ************** training configuration **************
parser.add_argument("-random_seed", type=int, default=19960804, help="Random seed for tensorflow and numpy")
parser.add_argument("-batch_size", type=int, default=32, help="batch size")
parser.add_argument("-optimizer", type=str, default="adam", help="optimizer (or, trainer)")
parser.add_argument("-lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("-n_epoch", type=int, default=35, help="number of training epoch")
parser.add_argument("-adapt_rate", type=float, default=0.1, help="adapt rate")
parser.add_argument("-adv_weight", type=float, default=1.0, help="adversarial weight")
parser.add_argument("-dropout_rate", type=float, default=0.5, help="dropout rate")
parser.add_argument("-clip_grad", type=float, default=40.0, help="maximum gradients")
parser.add_argument("-lr_decay", type=float, default=0.05, help="decay rate of learning rate")
parser.add_argument("-evaluation_interval", type=int, default=1, help="Evaluate and print results every x epochs")

# ************** network configuration **************
parser.add_argument("-input_win", type=int, default=3, help="window size of input")
parser.add_argument("-stm_win", type=int, default=3, help="window size of OE component")
parser.add_argument("-dim_lm_y", type=int, default=2, help="class number for the lm module")
parser.add_argument("-dim_rel", type=int, default=50, help="dimension of latent relations between global memory and local memory")
parser.add_argument("-dim_asp_h", type=int, default=50, help="hidden dimension for aspect extraction (aspect boundary tags)")
parser.add_argument("-dim_opn_h", type=int, default=50, help="hidden dimension for joint extraction  (unified tags)")
parser.add_argument("-dim_ts_h", type=int, default=50, help="hidden dimension for joint extraction  (unified tags)")
parser.add_argument("-hops", type=int, default=2, help="the number of the hops for recurrent attention.")

# ************** preprocessing configuration **************
parser.add_argument("-emb_name", '-e',type=str, default="yelp_electronics", help="name of word embedding")
parser.add_argument("-tagging_schema", type=str, default="BIEOS", help="tagging schema")
parser.add_argument("-selection_schema", type=str, default="TS", help="tagging schema")

args = parser.parse_args()

emb2path = {
    './data/w2v_merge_norm100_10.txt'
}

pair2epoch = {
    'rest-laptop': 30,
    'rest-device': 25,
    'device-rest': 40,
    'laptop-rest': 35,

    'service-rest': 50,
    'service-device': 65,
    'service-laptop': 70,

    'rest-service': 30,
    'device-service': 60,
    'laptop-service': 45
}

pair2schema = {
    'rest-laptop': 'OTE_TS',
    'rest-device': 'OTE_TS',
    'device-rest': 'OTE_TS',
    'laptop-rest': 'OTE_TS',

    'service-rest': 'TS',
    'service-device': 'TS',
    'service-laptop': 'TS',

    'rest-service': 'TS',
    'device-service': 'TS',
    'laptop-service': 'TS'
}
