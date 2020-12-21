import argparse

parser=argparse.ArgumentParser(description="Onion Peel Network")

parser.add_argument('--project_name',type=str,default="video completion by Onion Peel Network",
                    help='工程名')
parser.add_argument("--use_cuda",type=bool,default=True,
                    help="是否想使用cuda")
parser.add_argument("--seed",type=int,default=123,
                    help="随机种子")
parser.add_argument("--resume",type=bool,default=True,
                    help="是否使用预训练的权重加载模型")
parser.add_argument("--pretrained_weight",type=str,default='OPN.pth',
                    help="预训练模型加载路径")
parser.add_argument("--lr",type=float,default=0.0001,
                    help="学习率")
parser.add_argument("--weight_decay",type=float,default=1e-4,
                    help="权重衰减系数")
parser.add_argument("--momentum",type=float,default=0.5,
                    help="动量系数")
parser.add_argument("--epoch",type=int,default=10,
                    help="训练epoch次数")
parser.add_argument("--train_batch_size",type=int,default=1,
                    help="训练batch_size")
parser.add_argument("--test_batch_size",type=int,default=1,
                    help="测试batch_szie")
parser.add_argument("--save",type=bool,default=True,
                    help="保存图片")
