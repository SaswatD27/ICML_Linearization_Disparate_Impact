import os
import time
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.datasets as dst
from models.utils import load_model
import torchvision.transforms as transforms


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# =============================================================================#
#                                                                             #
#                    ████████╗██████╗  █████╗ ██╗███╗   ██╗                   #
#                    ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║                   #
#                       ██║   ██████╔╝███████║██║██╔██╗ ██║                   #
#                       ██║   ██╔══██╗██╔══██║██║██║╚██╗██║                   #
#                       ██║   ██║  ██║██║  ██║██║██║ ╚████║                   #
#                       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝                   #
#                                                                             #
# =============================================================================#


def train(train_loader, nets, optimizer, criterions, epoch, lambda_kd, kd_mode, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cls_losses = AverageMeter()
    kd_losses = AverageMeter()
    cummulative_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets["snet"]
    tnet = nets["tnet"]

    criterionCls = criterions["criterionCls"]
    criterionKD = criterions["criterionKD"]

    snet.train()

    end = time.time()
    for i, (img, target) in enumerate(train_loader, start=1):
        target = target.squeeze(dim=1).long()
        #print('Image Shape: ',img.shape)
        #target = target.type(torch.LongTensor)
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            img = img.to(device)
            target = target.to(device)
            snet = snet.to(device)
            tnet = tnet.to(device)

        out_s = snet(img)
        out_t = tnet(img)
        #print(f'output: {out_s.shape}')
        #print(f'target: {target.shape}')

        cls_loss = criterionCls(out_s, target) * lambda_kd
        if kd_mode in ["st"]:
            kd_loss = criterionKD(out_s, out_t.detach()) * (1 - lambda_kd)
        else:
            raise Exception("Invalid kd mode...")

        loss = cls_loss + kd_loss

        prec1, prec5 = accuracy(out_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        kd_losses.update(kd_loss.item(), img.size(0))
        cummulative_loss.update(loss, img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    return cls_losses.avg, kd_losses.avg, cummulative_loss.avg, top1.avg, top5.avg


# =============================================================================#
#                                                                             #
#                      ████████╗███████╗███████╗████████╗                     #
#                      ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝                     #
#                         ██║   █████╗  ███████╗   ██║                        #
#                         ██║   ██╔══╝  ╚════██║   ██║                        #
#                         ██║   ███████╗███████║   ██║                        #
#                         ╚═╝   ╚══════╝╚══════╝   ╚═╝                        #
#                                                                             #
# =============================================================================#


def test(test_loader, nets, criterions, epoch, kd_mode, lambda_kd, device):
    cls_losses = AverageMeter()
    kd_losses = AverageMeter()
    cummulative_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets["snet"]
    tnet = nets["tnet"]

    criterionCls = criterions["criterionCls"]
    criterionKD = criterions["criterionKD"]

    snet.eval()

    end = time.time()
    for i, (img, target) in enumerate(test_loader, start=1):
        target = target.squeeze(dim=1).long()
        if torch.cuda.is_available():
            img = img.to(device)
            target = target.to(device)
            snet = snet.to(device)
            tnet = tnet.to(device)

        with torch.no_grad():
            out_s = snet(img)
            out_t = tnet(img)

        cls_loss = criterionCls(out_s, target)
        if kd_mode in ["logits", "st"]:
            kd_loss = criterionKD(out_s, out_t.detach()) * lambda_kd
        else:
            raise Exception("Invalid kd mode...")

        loss = cls_loss + kd_loss

        prec1, prec5 = accuracy(out_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        kd_losses.update(kd_loss.item(), img.size(0))
        cummulative_loss.update(loss, img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    return cls_losses.avg, kd_losses.avg, cummulative_loss.avg, top1.avg, top5.avg


class SoftTarget(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''

    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                        F.softmax(out_t / self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss


def train_DeepReduce(args,
                    trainloader,
                    testloader,
                    device,
                    logger,
                    tnet, 
                    snet,
                    ):
    # Model factory..
    logger.info('==> Building model..')
    #snet = load_model(args.net, args).to(device)
    #tnet = load_model(args.tnet, args).to(device)

    # tnet_model = torch.load(args.tnet_path)["net"]
    # remove the substring 'module.' from the keys
    # tnet_model = {k[7:]: v for k, v in tnet_model.items()}
    # tnet.load_state_dict(tnet_model)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        snet = nn.DataParallel(snet).to(device)
        tnet = nn.DataParallel(tnet).to(device)

    snet.train()
    tnet.eval()

    if args.kd_mode == "st":
        criterionKD = SoftTarget(args.temperature)
    else:
        raise Exception("Invalid kd mode...")

    if torch.cuda.is_available():
        criterionCls = torch.nn.CrossEntropyLoss().to(device)
    else:
        criterionCls = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        snet.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    if "resnet" in args.net.lower():
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        # 	optimizer, milestones=[30, 60, 90], gamma=0.1
        # )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200)

        # define transforms
    # if args.dataset.lower() == "cifar100":
    #     dataset = dst.CIFAR100
    #     mean = (0.5071, 0.4865, 0.4409)
    #     std = (0.2673, 0.2564, 0.2762)
    #
    # elif args.dataset.lower() == "cifar10":
    #     dataset = dst.CIFAR10
    #     mean = (0.5071, 0.4865, 0.4409)
    #     std = (0.2673, 0.2564, 0.2762)
    #
    # if args.dataset.lower() in ['cifar100', 'cifar10']:
    #     train_transform = transforms.Compose(
    #         [
    #             transforms.Pad(4, padding_mode="reflect"),
    #             transforms.RandomCrop(32),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=mean, std=std),
    #         ]
    #     )
    #     test_transform = transforms.Compose(
    #         [
    #             transforms.CenterCrop(32),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=mean, std=std),
    #         ]
    #     )
    #
    #     # define data loader
    #     train_loader = torch.utils.data.DataLoader(
    #         dataset(
    #             root=path_dataset, transform=train_transform, train=True, download=True
    #         ),
    #         batch_size=batch_size,
    #         shuffle=True,
    #         num_workers=16,
    #         pin_memory=True,
    #     )
    #     test_loader = torch.utils.data.DataLoader(
    #         dataset(
    #             root=path_dataset, transform=test_transform, train=False, download=True
    #         ),
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=16,
    #         pin_memory=True,
    #     )
    #
    # else:
    #     raise ValueError(f"Invalid dataset")

    # warp nets and criterions for train and test
    nets = {"snet": snet, "tnet": tnet}
    criterions = {"criterionCls": criterionCls, "criterionKD": criterionKD}

    best_top1 = 0
    best_top5 = 0

    progress_bar = tqdm(range(0, args.n_epochs))

    for epoch in progress_bar:

        # train one epoch
        epoch_start_time = time.time()
        tr_cls_loss, tr_kd_loss, tr_cummu_loss, tr_top1, tr_top5 = train(
            trainloader, nets, optimizer, criterions, epoch, lambda_kd=args.lambda_kd, kd_mode=args.kd_mode,
            device=device
        )

        # evaluate on testing set
        test_cls_loss, test_kd_loss, test_cummu_loss, test_top1, test_top5 = test(
            testloader, nets, criterions, epoch, lambda_kd=args.lambda_kd, kd_mode=args.kd_mode, device=device
        )

        epoch_duration = time.time() - epoch_start_time

        if "wideresnet" in args.net:
            scheduler.step()
        elif "resnet" in args.net:
            scheduler.step()
        else:
            scheduler.step()
            # adjust_lr(optimizer, epoch, lr, lr_stepsize)

        temp_lr = None
        for param_group in optimizer.param_groups:
            temp_lr = param_group["lr"]

        # training_losses = [type(x) for x in [tr_cls_loss, tr_kd_loss, tr_cummu_loss]]
        # print(training_losses)
        progress_bar.set_description(
            f"Ep: {epoch} eptime: {epoch_duration:.2f} TrLosses [CLS, KD, TOTAL]: {tr_cls_loss:.3}, {tr_kd_loss:.3}, {tr_cummu_loss:.3}  TrTop1: {tr_top1:.2f} TrTop5: {tr_top5:.2f}  EvLosses [CLS, KD, TOTAL]: {test_cls_loss:.2}, {test_kd_loss:.2}, {test_cummu_loss:.2f}  EvTop1: {test_top1:.2f}  EvTop5: {test_top5:.2f} lr: {temp_lr:.5f}"
        )
        progress_bar.refresh()

        # save model
        is_best = False
        if test_top1 > best_top1:
            best_top1 = test_top1
            best_top5 = test_top5
            is_best = True

            state = {"model": snet.state_dict(),
                    "model_arch": args.net}

            torch.save(state, args.savepath)
