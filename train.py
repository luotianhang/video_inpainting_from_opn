import os
import sys

import torch.backends.cudnn as cudnn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from common import *
from datalist import Dataset2
from models.OPN import OPN
from utils.helpers import *

sys.path.append('utils/')
sys.path.append('models/')

style_weights = {
    'conv1_1': 1,
    'conv2_1': 0.8,
    'conv3_1': 0.5,
    'conv4_1': 0.3,
    'conv5_1': 0.1,
}
from config import parser


class train(object):
    def __init__(self):
        self.args = parser.parse_args()
        print(f"-----------{self.args.project_name}-----------")
        use_cuda = self.args.use_cuda and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(self.args.seed)
        else:
            torch.manual_seed(self.args.seed)

        self.device = torch.device("cuda" if use_cuda else "cpu")

        kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

        '''
        构造DataLoader
        '''
        # ToDo 数据集需要重新制备
        print("Create Dataloader")
        self.train_loader = DataLoader(Dataset2(), batch_size=1, shuffle=True, **kwargs)
        self.test_loader = DataLoader(Dataset2(), batch_size=1, shuffle=True, **kwargs)
        '''
        定义模型
        '''
        print("Create Model")
        self.model = OPN().to(self.device)
        self.model = nn.DataParallel(OPN())
        if use_cuda:
            # self.model = self.model.cuda()
            cudnn.benchmark = True
        '''
        根据需要加载预训练的模型权重参数
        '''

        # VGG16模型配合预训练的模型用于检测
        self.vgg = models.vgg19(pretrained=True).to(self.device).features

        for i in self.vgg.parameters():
            i.requires_grad = False
        try:
            if self.args.resume and self.args.pretrained_weight:
                self.model.load_state_dict(torch.load(os.path.join('OPN.pth')), strict=False)
                print("模型加载成功")
        except:
            print("模型加载失败")
        '''
        cuda加速
        '''
        if use_cuda:
            self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        '''
        构造loss目标函数
        选择优化器
        学习率变化选择
        '''
        print("Establish the loss, optimizer and learning_rate function")
        self.loss_tv = TVLoss()
        # 另外还有style—loss 和 content—loss
        # self.optimizer = optim.SGD(
        #     params=self.model.parameters(),
        #     lr=self.args.lr,
        #     weight_decay=self.args.weight_decay,
        #     momentum=0.5
        # )
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,  # 为了防止分母为0
            weight_decay=0
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=1e-5)
        '''
        模型开始训练
        '''
        print("Start training")
        for epoch in tqdm(range(self.args.epoch + 1)):
            # self.train(epoch)
            # if epoch % 20 == 0:
                self.test(epoch)

        torch.cuda.empty_cache()

        print("finish model training")

    def train(self, epoch):
        self.model.train()

        for data in self.train_loader:
            self.optimizer.zero_grad()
            self.content_loss = 0
            self.style_loss = 0

            midx = list(range(0, 5))
            # frames被破损的图像，valids可获取的像素区域，dists填补的像素区域
            frames, valids, dists, label = data
            frames, valids, dists, label = frames.to(self.device), valids.to(self.device), dists.to(
                self.device), label.to(self.device)
            # 每一张图片都被encoder过了获得的key和val shape为（1，128，5，60，106），hol为（1，1，5，60，106）
            mkey, mval, mhol = self.model(frames[:, :, midx], valids[:, :, midx], dists[:, :, midx])

            loss = 0

            for f in range(5):
                # 对每张图取其他4张图作为reference的参考
                ridx = [i for i in range(len(midx)) if i != f]
                fkey, fval, fhol = mkey[:, :, ridx], mval[:, :, ridx], mhol[:, :, ridx]
                # 图像补全
                for r in range(999):
                    if r == 0:
                        # 取主图
                        comp = frames[:, :, f]
                        dist = dists[:, :, f]
                    # comp是破损的图片，逐层补全图片
                    # valids是没有缺失信息的区域
                    # dist是缺失信息的区域
                    '''
                    按dist的指导，逐8个像素的距离，循环修复图片，其中valids表示空洞部分的区域（0，1）
                    comp是在frame的基础之上补充的，相似度极高，只计算这一部分的loss
                    '''
                    comp, dist, peel = self.model(fkey, fval, fhol, comp, valids[:, :, f], dist)
                    # 每次循环中分别在像素空间和深层特征空间最小化和GT的L1距离。
                    loss += 100 * L1(comp, label[:, :, f], peel)
                    loss += L1(comp, label[:, :, f], valids[:, :, f])

                    # loss+=100*ll1(comp,frames[:,:,f])

                    # content loss
                    content_features = get_features(frames[:, :, f], self.vgg)
                    target_features = get_features(comp, self.vgg)
                    self.content_loss = torch.mean(
                        torch.abs((target_features['conv4_2'] - content_features['conv4_2'])))
                    loss = loss + 0.05 * self.content_loss
                    # style loss
                    style_features = get_features(comp, self.vgg)
                    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
                    '''加上每一层的gram_matrix矩阵的损失'''
                    for layer in style_weights:
                        target_feature = target_features[layer]
                        target_gram = gram_matrix(target_feature)
                        _, d, h, w = target_feature.shape
                        style_gram = style_grams[layer]
                        layer_style_loss = style_weights[layer] * torch.mean(torch.abs((target_gram - style_gram)))
                        self.style_loss += layer_style_loss / (d * h * w)  # 加到
                    loss = loss + 120 * self.style_loss

                    if torch.sum(dist).item() == 0:
                        break

                # tv loss
                loss += 0.01 * self.loss_tv(comp)
            loss.backward()
            self.optimizer.step()
        print("epoch{}".format(epoch) + "  loss:{}".format(loss.cpu()))

    def test(self, epoch):
        self.model.eval()
        for frames, valids, dists, label in self.test_loader:
            midx = list(range(0, 5))
            # frames, valids, dists = data
            frames, valids, dists = frames.to(self.device), valids.to(self.device), dists.to(self.device)
            with torch.no_grad():
                # 先把这5张图片都encoder一下
                mkey, mval, mhol = self.model(frames[:, :, midx], valids[:, :, midx], dists[:, :, midx])
            # 对每张图取其他4张图作为reference的参考
            for f in range(5):
                ridx = [i for i in range(len(midx)) if i != f]
                fkey, fval, fhol = mkey[:, :, ridx], mval[:, :, ridx], mhol[:, :, ridx]
                # 图像补全
                for r in range(999):
                    if r == 0:
                        comp = frames[:, :, f]
                        dist = dists[:, :, f]
                    with torch.no_grad():
                        comp, dist, peel = self.model(fkey, fval, fhol, comp, valids[:, :, f], dist)

                    comp, dist = comp.detach(), dist.detach()
                    # 空隙填满进入后，把图片保存，然后进入下一轮图片的计算过程中
                    if torch.sum(dist).item() == 0:
                        break

                if self.args.save:

                    # visualize..
                    est = (comp[0].permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)
                    true = (label[0, :, f].permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)  # h,w,3
                    # mask = (dists[0, 0, f].detach().cpu().numpy() > 0).astype(np.uint8)  # h,w,1
                    # ov_true = overlay_davis(true, colors=[[0, 0, 0], [100, 100, 0]], cscale=2, alpha=0.4)

                    canvas = np.concatenate([true,est], axis=0)
                    save_path = os.path.join('Results')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    canvas = Image.fromarray(canvas)
                    canvas.save(os.path.join(save_path, 'res_{}_{}.jpg'.format(epoch, f)))

        print("epoch{}".format(epoch) + " test finished")


if __name__ == "__main__":
    train()
