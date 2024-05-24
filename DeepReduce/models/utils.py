from models import resnet_deepreduce_version as ResNetDR

def load_model(net_name, args):
    assert(args.thinned in [0,1])
    if args.thinned == 0:
        thinned_val = False
    elif args.thinned == 1:
        thinned_val = True
    '''
    print('WITHIN LOAD_MODEL')
    print(f'\n NET: {args.net, type(args.net)}\n')
    print(f'\n CULLED_MASK: {args.culled_mask, type(args.culled_mask)}\n')
    print(f'\n THINNED: {args.thinned, type(args.thinned)}\n')
    print(f'\n ALPHA: {args.alpha, type(args.alpha)}\n')
    print(f'\n SEED: {args.seed, type(args.seed)}\n')
    assert(args.thinned in [0,1])
    if args.thinned == 0:
        thinned_val = False
    elif args.thinned == 1:
        thinned_val = True
    print(f'thinned status: {thinned_val}')
    '''
    print(f'thinned status: {thinned_val}')
    if net_name == 'resnet18_deepreduce_teacher_cifar10':
        net = ResNetDR.resnet18(groups=0, groupsize=-1, residual=True, num_classes=10,
                                Culled=0, Thinned=False, alpha=1.0, rho=1.0)
    elif net_name == 'resnet18_deepreduce_student_cifar10':
        net = ResNetDR.resnet18(groups=0, groupsize=-1, residual=True, num_classes=10,
                                Culled=args.culled_mask, Thinned=thinned_val, alpha=args.alpha, rho=args.rho)
    elif net_name == 'resnet34_deepreduce_student_cifar10':
        net = ResNetDR.resnet34(groups=0, groupsize=-1, residual=True, num_classes=10,
                                Culled=args.culled_mask, Thinned=thinned_val, alpha=args.alpha, rho=args.rho)
    elif net_name == 'resnet34_deepreduce_teacher_cifar10':
        net = ResNetDR.resnet34(groups=0, groupsize=-1, residual=True, num_classes=10,
                                Culled=0, Thinned=False, alpha=1.0, rho=1.0)
    elif net_name == 'kundu_cifar10':
        net = load_kundu_model(model_name='CustomResNet18', n_cls=10)
    else:
        raise NotImplementedError()
    return net
