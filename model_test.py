from __future__ import division

import argparse
import os

import cv2
from PIL import Image

from models.OPN import OPN
from utils.helpers import *


def get_arguments():
    parser = argparse.ArgumentParser(description='demasking')
    parser.add_argument("--input", type=str, default='H', required=False)
    return parser.parse_args()


# 模型初始化
model = nn.DataParallel(OPN())
if torch.cuda.is_available():
    model.cuda()
model.load_state_dict(torch.load(os.path.join('OPN.pth')), strict=False)
model.eval()

args = get_arguments()
seq_name = args.input

# T=len(os.listdir('./Image_inputs/'+seq_name))
# 基本参数初始化
T = 5
H, W = cv2.imread('./Video_inputs/' + seq_name + '/0000.jpg').shape[0], \
       cv2.imread('./Video_inputs/' + seq_name + '/0000.jpg').shape[1]

length = len(os.listdir('./Video_inputs/' + seq_name))

start_point = 0
for j in range(length // T-1):
    frames = np.empty((T, H, W, 3), dtype=np.float32)
    holes = np.empty((T, H, W, 1), dtype=np.float32)
    dists = np.empty((T, H, W, 1), dtype=np.float32)
    # 获取图片
    for i in range(start_point, start_point + T):
        img_file = os.path.join('Video_inputs', seq_name, '{:04d}.jpg'.format(i))
        raw_frame = np.array(Image.open(img_file).convert('RGB')) / 255.
        raw_frame = cv2.resize(raw_frame, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
        frames[i - start_point] = raw_frame

        mask_file = os.path.join('Video_inputs', seq_name, '{:04d}.png'.format(i))
        raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
        raw_mask = (raw_mask > 0.5).astype(np.uint8)
        raw_mask = cv2.resize(raw_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        raw_mask = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        holes[i - start_point, :, :, 0] = raw_mask.astype(np.float32)
        dists[i - start_point, :, :, 0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2, maskSize=5)

    frames = torch.from_numpy(np.transpose(frames, (3, 0, 1, 2)).copy()).float()
    holes = torch.from_numpy(np.transpose(holes, (3, 0, 1, 2)).copy()).float()
    dists = torch.from_numpy(np.transpose(dists, (3, 0, 1, 2)).copy()).float()

    frames = frames * (1 - holes) + holes * torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    valids = 1 - holes

    frames = frames.unsqueeze(0)
    holes = holes.unsqueeze(0)
    dists = dists.unsqueeze(0)
    valids = valids.unsqueeze(0)

    midx = list(range(0, T))
    with torch.no_grad():
        mkey, mval, mhol = model(frames[:, :, midx], valids[:, :, midx], dists[:, :, midx])

    for f in range(T):
        ridx = [i for i in range(len(midx)) if i != f]
        fkey, fval, fhol = mkey[:, :, ridx], mval[:, :, ridx], mhol[:, :, ridx]
        for r in range(999):
            if r == 0:
                comp = frames[:, :, f]
                dist = dists[:, :, f]
            with torch.no_grad():
                comp, dist = model(fkey, fval, fhol, comp, valids[:, :, f], dist)

            comp, dist = comp.detach(), dist.detach()
            if torch.sum(dist).item() == 0:
                break

        est = (comp[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        save_path = os.path.join("Image_results", seq_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        est = Image.fromarray(est)
        est.save(os.path.join(save_path, 'result_{}.jpg'.format(f + start_point)))
    start_point += T

print('Results are saved: ./{}'.format(save_path))
