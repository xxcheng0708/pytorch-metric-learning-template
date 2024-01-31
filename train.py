import os
import torch
from utils.data_prefetcher import DataPrefetcher
from utils.mixup import mixup_data
from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
import warnings
from utils.evaluate import get_all_embeddings, do_metric, do_validation_loss

warnings.filterwarnings("ignore")


def train(model, train_iter, train_set, val_iter, val_set, loss_func_list, num_epoches, accuracy_calculator, save_dir,
          optimizer=None, loss_optimizer=None, mining_func=None, writer=None, knn_model=None,
          mixup_enable=False, xbm_enable=False, xbm_start_iteration=0, xbm_close=0, validate_loss=True,
          ref_includes_query=False, print_step=10):
    lr_decay_list = []
    # 学习率衰减
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # lr_scheduler = CosineAnnealingLRWarmup(optimizer, T_max=num_epoches, warmup_start_lr=1.0e-6,
    #                                        warmup_steps=5, last_epoch=-1)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10,
                                                   num_training_steps=len(train_iter) * num_epoches)
    # lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=10,
    #                                                                   num_training_steps=len(train_iter) * num_epoches,
    #                                                                   num_cycles=2)

    print("xbm enable: {}, xbm iteration: {}, mixup enable: {}".format(xbm_enable, xbm_start_iteration, mixup_enable))
    print(loss_func_list[0], loss_func_list[1], mining_func)

    loss_func = loss_func_list[0]
    for epoch in range(num_epoches):
        # lr_scheduler.step()
        model.train()
        train_loss_sum = 0.0
        n = 0

        # 进行训练数据预加载，提升数据加载效率
        prefetcher = DataPrefetcher(train_iter)
        X, y = prefetcher.next()
        batch_idx = 0
        while X is not None:
            if xbm_enable:
                # xbm_start_iteration轮迭代之后，使用 xbm 跨批量增强
                if epoch * len(train_iter) + batch_idx >= xbm_start_iteration:
                    loss_func = loss_func_list[1]
            else:
                loss_func = loss_func_list[0]

            # 提前结束 xbm 跨批量增强
            if epoch > num_epoches - xbm_close:
                xbm_enable = False

            X = X.cuda()
            y = y.cuda()
            # print("train: {}, {}".format(X.shape, y.shape))
            # print(y)

            if mixup_enable:
                # 使用 mixup 数据增强
                X, y_a, y_b, lam = mixup_data(X, y, 0.1, True)
                X, y_a, y_b = map(torch.tensor, (X, y_a, y_b))
                X = X.cuda()
                y_a = y_a.cuda()
                y_b = y_b.cuda()

                y_pred = model(X)
                # y_pred = torch.nn.functional.normalize(y_pred, p=2, dim=1)
                loss = lam * loss_func(y_pred, y_a) + (1 - lam) * loss_func(y_pred, y_b)
            else:
                y_pred = model(X)
                # y_pred = torch.nn.functional.normalize(y_pred, p=2, dim=1)
                if mining_func:
                    # 在 batch 中进行难训练样本挖掘，提升效果
                    indices_tuple = mining_func(y_pred, y)
                    loss = loss_func(y_pred, y, indices_tuple)
                else:
                    loss = loss_func(y_pred, y)

            loss = torch.mean(loss)

            optimizer.zero_grad()
            if loss_optimizer:
                loss_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss_optimizer:
                loss_optimizer.step()

            train_loss_sum += loss.item()
            n += y.shape[0]

            if batch_idx % print_step == 0:
                print("Epoch {} Iteration {}, Loss = {}".format(epoch, batch_idx, loss))
                writer.add_scalar("loss/iter", loss.item(), epoch * len(train_iter) + batch_idx)

            model.train()
            X, y = prefetcher.next()

            lr_decay_list.append(optimizer.state_dict()["param_groups"][0]["lr"])
            writer.add_scalar("learning rate", lr_decay_list[-1], epoch * len(train_iter) + batch_idx)
            lr_scheduler.step()

            batch_idx += 1

        print("Epoch {}, Train Loss = {}".format(epoch, train_loss_sum / n))
        writer.add_scalar("loss/train", train_loss_sum / n, epoch)

        with torch.no_grad():
            if validate_loss:
                val_loss = do_validation_loss(val_iter, model, loss_func_list[0], mining_func, print_step)
                print("Epoch {}, Val Loss ={}".format(epoch, val_loss))
                writer.add_scalar("loss/val", val_loss, epoch)

            if ref_includes_query is False:
                acc = do_metric(train_set, val_set, model, accuracy_calculator, knn_model,
                                ref_includes_query=ref_includes_query)
            else:
                acc = do_metric(val_set, val_set, model, accuracy_calculator, knn_model,
                                ref_includes_query=ref_includes_query)
            precision_at_1 = acc["precision_at_1"]
            r_precision = acc["r_precision"]
            r_map = acc["mean_average_precision_at_r"]
            knn_score = acc["knn_score"]

            writer.add_scalar("accuracy/precision@1", precision_at_1, epoch)
            writer.add_scalar("accuracy/r_precision", r_precision, epoch)
            writer.add_scalar("accuracy/mean_average_precision_at_r", r_map, epoch)
            writer.add_scalar("accuracy/knn_score", knn_score, epoch)

            best_model_name = os.path.join(save_dir, "model-acc-%03d-%.04f-%.04f-%.04f.pth" % (
                epoch, r_map, precision_at_1, r_precision))
            torch.save(model.module.state_dict(), best_model_name)
