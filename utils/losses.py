from pytorch_metric_learning import losses, distances, miners
import torch
from pytorch_metric_learning.utils import common_functions as c_f

c_f.COLLECT_STATS = True


def get_loss(loss_name, xbm_enable, lr, weight_decay, weight_regularizer=None, classes_num=0, out_dimension=128, xbm_size=10240):
    distance = None
    loss_miner = None
    loss_func = None
    xbm_loss_func = None
    loss_optimizer = None
    distance = None

    ### Paid Based Loss
    if loss_name == "TripletMarginLoss":
        # TripletMarginLoss
        distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
        loss_miner = miners.TripletMarginMiner(margin=0.8, distance=distance, type_of_triplets="semihard")
        loss_func = losses.TripletMarginLoss(margin=0.5, distance=distance, swap=True)
        xbm_loss_func = None
        if xbm_enable:
            xbm_loss_func = get_xbm_loss_func(loss_func, xbm_size, out_dimension, loss_miner)
    if loss_name == "MultiSimilarityLoss":
        # MultiSimilarityLoss
        distance = distances.CosineSimilarity()
        loss_miner = miners.MultiSimilarityMiner(distance=distance)
        loss_func = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5, distance=distance)
        if xbm_enable:
            xbm_loss_func = get_xbm_loss_func(loss_func, xbm_size, out_dimension, loss_miner)

    if loss_name == "ContrastiveLoss":
        # ContrastiveLoss
        distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
        loss_miner = miners.PairMarginMiner(pos_margin=0.1, neg_margin=1.0, distance=distance)
        loss_func = losses.ContrastiveLoss(distance=distance, pos_margin=0.1, neg_margin=1.0)
        if xbm_enable:
            xbm_loss_func = get_xbm_loss_func(loss_func, xbm_size, out_dimension, loss_miner)
    if loss_name == "CircleLoss":
        # Circle Loss
        distance = distances.CosineSimilarity()
        loss_miner = miners.MultiSimilarityMiner(distance=distance)
        # face recognition
        # loss_func = losses.CircleLoss(m=0.25, gamma=256, distance=distance)
        # fine-grained ime retrieval
        loss_func = losses.CircleLoss(m=0.4, gamma=80, distance=distance)
        if xbm_enable:
            xbm_loss_func = get_xbm_loss_func(loss_func, xbm_size, out_dimension, loss_miner)

    if loss_name == "SupervisedContrastiveLoss":
        # SupervisedContrastiveLoss
        distance = distances.CosineSimilarity()
        loss_miner = miners.MultiSimilarityMiner(distance=distance)
        loss_func = losses.SupConLoss(temperature=0.1, distance=distance)
        if xbm_enable:
            xbm_loss_func = get_xbm_loss_func(loss_func, xbm_size, out_dimension, loss_miner)

    ### CrossEntropy Loss(requires an loss optimizer)
    if loss_name == "ArcFaceLoss":
        # ArcFaceLoss
        distance = distances.CosineSimilarity()
        loss_miner = miners.MultiSimilarityMiner(distance=distance)
        loss_func = losses.ArcFaceLoss(num_classes=classes_num, embedding_size=out_dimension, margin=0.5, scale=64,
                                       distance=distance)

    if loss_name == "ProxyAnchorLoss":
        # ProxyAnchorLoss
        distance = distances.CosineSimilarity()
        loss_miner = miners.MultiSimilarityMiner(distance=distance)
        loss_func = losses.ProxyAnchorLoss(num_classes=classes_num, embedding_size=out_dimension, margin=0.1, alpha=32,
                                           distance=distance)

    if loss_name == "SoftTripletLoss":
        # SoftTripletLoss
        distance = distances.CosineSimilarity()
        loss_miner = miners.MultiSimilarityMiner(distance=distance)
        loss_func = losses.SoftTripleLoss(num_classes=classes_num, embedding_size=out_dimension, distance=distance)

    if loss_name == "LargeMarginSoftmaxLoss":
        # LargeMarginSoftmaxLoss
        distance = distances.CosineSimilarity()
        loss_miner = miners.MultiSimilarityMiner(distance=distance)
        loss_func = losses.LargeMarginSoftmaxLoss(num_classes=classes_num, embedding_size=out_dimension,
                                                  distance=distance)

    if classes_num != 0:
        loss_optimizer = torch.optim.SGD(
            loss_func.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    return loss_func, xbm_loss_func, loss_miner, loss_optimizer, distance


def get_xbm_loss_func(loss_func, xbm_size, out_dimension, loss_miner=None):
    xbm_loss_func = losses.CrossBatchMemory(loss=loss_func, embedding_size=out_dimension,
                                            memory_size=xbm_size,
                                            miner=loss_miner)
    return xbm_loss_func
