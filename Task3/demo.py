import torch
import numpy as np
import cv2
import imutils


from torch.utils.data import DataLoader
from cityscapes import CityScapes
from model import model
from arg_parser import demo


def main(args):
    scale = 1
    cropsize = [int(2048 * scale), int(1024 * scale)]

    ds = CityScapes(args.cityscapes_path, cropsize=cropsize, mode='val', demo=True)
    n_classes = ds.n_classes

    dl = DataLoader(ds,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=True)

    net = model.get_network(n_classes)

    saved_path = args.saved_model
    loaded_model = torch.load(saved_path, map_location=torch.device('cuda') if torch.cuda.is_available() else 'cpu')

    state_dict = loaded_model['state_dict']
    net.load_state_dict(state_dict, strict=False)

    if torch.cuda.is_available():
        net.cuda()

    net.eval()

    for images, im, lb in dl:
        with torch.no_grad():
            images = images.numpy()
            lb = lb.numpy()

            if torch.cuda.is_available():
                im = im.cuda()

            preds = net(im).argmax(dim=1).cpu().numpy()

            for image, pred, label in zip(images, preds, lb):
                label = ds.vis_label(label)
                pred = ds.vis_label(pred)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('im', imutils.resize(cv2.hconcat([image, label, pred]), width=1920))

                if ord('q') == cv2.waitKey(0):
                    exit()
    

if __name__ == '__main__':
    args = demo()
    main(args)
