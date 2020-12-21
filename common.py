import torch
import torch.nn as nn


def get_features(img, model, layers=None):
    '''获取特征层'''
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content层
            '28': 'conv5_1'
        }

    features = {}
    x = img
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    '''计算Gram matrix'''
    _, d, h, w = tensor.size()  # 第一个是batch_size

    tensor = tensor.view(d, h * w)

    gram = torch.mm(tensor, tensor.t())

    return gram


'''
TV loss是常用的一种正则项（注意是正则项，配合其他loss一起使用，约束噪声）
图片中相邻像素值的差异可以通过降低TV loss来一定程度上解决

图像上的一点点噪声可能就会对复原的结果产生非常大的影响，因为很多复原算法都会放大噪声。
这时候我们就需要在最优化问题的模型中添加一些正则项来

保持图像的光滑性
'''

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])  # 算出总共求了多少次差
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class L1_Loss(nn.Module):
    def __init__(self):
        super(L1_Loss, self).__init__()

    def forward(self, x, y, pv):
        loss = 0
        pv = pv
        if pv.ndim > 4:
            pv = pv
            pv = pv.squeeze(dim=0)
            for i in range(pv.size(1)):
                loss += torch.sum(torch.abs(x - y) * pv[:, i, :, :])
            return loss
        else:
            loss = torch.sum(torch.abs(x - y) * pv)
            return loss

class L1_Lossv2(nn.Module):
    def __init__(self):
        super(L1_Lossv2, self).__init__()

    def forward(self,x,y,pv):
        loss=0
        if pv.ndim>4:
            for i in range(pv.size()[1]):
                temp=pv[:,:,i]
                loss=loss+torch.sum((torch.abs(x.flatten()-y.flatten())*temp.flatten()))
            return loss
        else:
            loss=torch.sum(torch.abs(x.flatten()-y.flatten())*pv.flatten())
            return loss

def L1(x,y,mask):
    res=torch.abs(x-y)
    res=res*mask
    return torch.sum(res)

def ll1(x,y):
    return torch.sum(x-y)


