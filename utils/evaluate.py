from pytorch_metric_learning import testers
from sklearn.model_selection import cross_val_score
import torch


def get_all_embeddings(dataset, model):
    tester = testers.GlobalEmbeddingSpaceTester(normalize_embeddings=True)
    return tester.get_all_embeddings(dataset, model)


def do_metric(std_dataset, query_dataset, model, accuracy_calculator, knn_model=None, ref_includes_query=False):
    """
    通过计算特征向量之间的相似度，计算召回率指标
    :param std_dataset:
    :param query_dataset:
    :param model:
    :param accuracy_calculator:
    :param knn_model:
    :return:
    """
    torch.cuda.empty_cache()
    model.eval()
    std_embeddings, std_labels = get_all_embeddings(std_dataset, model)
    query_embeddings, query_labels = get_all_embeddings(query_dataset, model)

    std_labels = std_labels.squeeze(1)
    query_labels = query_labels.squeeze(1)

    # std_embeddings = std_embeddings.cpu()
    # std_labels = std_labels.cpu()
    # query_embeddings = query_embeddings.cpu()
    # query_labels = query_labels.cpu()

    accuracies = accuracy_calculator.get_accuracy(query_embeddings, query_labels,
                                                  std_embeddings, std_labels,
                                                  ref_includes_query=ref_includes_query)

    knn_score = 0.0
    if knn_model is not None:
        scores = cross_val_score(knn_model, query_embeddings, query_labels, cv=5, scoring="accuracy")
        knn_score = np.mean(scores)
        print("KNN accuracy: {}".format(knn_score))
    accuracies["knn_score"] = knn_score
    print(accuracies)
    return accuracies


def do_validation_loss(data_iter, net, loss_func, mining_func, print_step=10):
    """
    计算在验证集上的 loss
    :param data_iter:
    :param net:
    :param loss_func:
    :param mining_func:
    :return:
    """
    net.eval()
    loss_sum = 0.0
    n = 0
    for batch_idx, (X, y) in enumerate(data_iter):
        X = X.cuda()
        y = y.cuda()

        # print("evaluate: {}".format(X.shape))
        y_pred = net(X)
        # y_pred = torch.nn.functional.normalize(y_pred, p=2, dim=1)
        if mining_func:
            indices_tuple = mining_func(y_pred, y)
            loss = loss_func(y_pred, y, indices_tuple)
        else:
            loss = loss_func(y_pred, y)

        loss_sum += loss.item()
        n += y.shape[0]

        if batch_idx % print_step == 0:
            print("validation, Iteration {}: Loss = {}".format(batch_idx, loss))
    return loss_sum / n
