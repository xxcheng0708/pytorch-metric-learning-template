import torch


def convertBNtoSyncBN(module, process_group=None):
    '''Recursively replace all BN layers to SyncBN layer.

    Args:
        module[torch.nn.Module]. Network
    '''
    # print("#" * 80)
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        sync_bn = torch.nn.SyncBatchNorm(module.num_features, module.eps, module.momentum,
                                         module.affine, module.track_running_stats, process_group)
        sync_bn.running_mean = module.running_mean
        sync_bn.running_var = module.running_var
        if module.affine:
            sync_bn.weight = module.weight
            sync_bn.bias = module.bias
        return sync_bn
    else:
        for name, child_module in module.named_children():
            # print(name, child_module)
            # setattr(module, name) = convertBNtoSyncBN(child_module, process_group=process_group)
            setattr(module, name, convertBNtoSyncBN(child_module, process_group=process_group))
        return module


if __name__ == '__main__':
    import torchvision
    model = torchvision.models.resnet18(pretrained=False)
    # model_new = convertBNtoSyncBN(model)      # equal torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print(model_new)

    # print(model.bn1.weight, type(model.bn1.weight.clone().detach()))

    model_new = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print(model_new)