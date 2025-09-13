import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import numpy as np
from os import remove
import h5py
from math import ceil


def _cosine_warm_ratio(epoch, warm_epochs, init_ratio):
    """
    余弦从 init_ratio -> 1.0 的平滑调度。
    epoch 从 1 开始计。
    """
    warm_epochs = max(1, int(warm_epochs))
    if epoch >= warm_epochs:
        return 1.0
    # t in [0,1)
    t = max(0.0, float(epoch - 1) / float(warm_epochs))
    return float(init_ratio + (1.0 - init_ratio) * (0.5 - 0.5 * np.cos(np.pi * t)))


def train(opt, model, encoder_dim, device, dataset, criterion, optimizer,
          train_set, whole_train_set, whole_training_data_loader, epoch, writer):

    # 读取/默认超参（可在命令行或配置里覆盖） 
    loss_warm_epochs = int(getattr(opt, "lossWarmupEpochs", 5))
    loss_warm_init   = float(getattr(opt, "lossWarmupInit", 0.1))  # 0.3 -> 1.0
    neg_warm_epochs  = int(getattr(opt, "negCurriculumEpochs", 5))
    neg_warm_init    = float(getattr(opt, "negCurriculumInitRatio", 0.40))  # 40% -> 100%
    grad_clip_norm   = float(getattr(opt, "gradClipNorm", 1.0))  # <=0 表示关闭

    # 当 epoch 在 warm 区间：给 loss 一个拉低系数；给负样本个数一个上限比例
    loss_warm_ratio = _cosine_warm_ratio(epoch, loss_warm_epochs, loss_warm_init)
    neg_warm_ratio  = _cosine_warm_ratio(epoch, neg_warm_epochs,  neg_warm_init)

    epoch_loss = 0.0
    startIter = 1  # keep track of batch iter across subsets for logging

    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]

    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    for subIter in range(subsetN):
        print('====> Building Cache')
        model.eval()
        with h5py.File(train_set.cache, mode='w') as h5:
            pool_size = encoder_dim
            if getattr(opt, "pooling", "avg").lower() == 'adapnet':
                pool_size = opt.outDims
            h5feat = h5.create_dataset("features", [len(whole_train_set), pool_size], dtype=np.float32)

            # 推理态构建缓存
            with torch.no_grad():
                for iteration, (inp, indices) in tqdm(
                    enumerate(whole_training_data_loader, 1),
                    total=max(1, len(whole_training_data_loader) - 1),
                    leave=False
                ):
                    image_encoding = inp.to(device, non_blocking=True).float()
                    seq_encoding = model.pool(image_encoding)
                    h5feat[indices.detach().cpu().numpy(), :] = seq_encoding.detach().float().cpu().numpy()
                    del inp, image_encoding, seq_encoding

        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])

        training_data_loader = DataLoader(
            dataset=sub_train_set,
            num_workers=opt.threads,
            batch_size=opt.batchSize,
            shuffle=True,
            collate_fn=getattr(dataset, "collate_with_negs", dataset.collate_fn),  # 优先 7项，回退 5项
            pin_memory=not opt.nocuda
        )

        if device.type == "cuda":
            print('Allocated:', torch.cuda.memory_allocated())
            print('Reserved :', torch.cuda.memory_reserved())

        model.train()
        for iteration, batch in tqdm(
            enumerate(training_data_loader, startIter),
            total=len(training_data_loader), leave=False
        ):
            # 兼容 5项/7项两种 collate 结果
            if batch is None or batch[0] is None:
                continue

            if len(batch) == 7:
                # 7项: query, positives, negatives1, negatives2, negCounts1, negCounts2, indices
                query, positives, negatives1, negatives2, negCounts1, negCounts2, indices = batch
                negatives = torch.cat([negatives1, negatives2], dim=0)
                negCounts = (negCounts1 + negCounts2).to(torch.long)
            elif len(batch) == 5:
                # 5项: query, positives, negatives1, negatives2, indices
                query, positives, negatives1, negatives2, indices = batch
                Btmp = int(query.shape[0])
                if negatives1.dim() < 2 or negatives2.dim() < 2:
                    raise ValueError("Expected negatives1/negatives2 to have at least 2 dims [B, n, ...].")
                n1 = int(negatives1.shape[1])
                n2 = int(negatives2.shape[1])
                neg1_flat = negatives1.reshape(Btmp * n1, *negatives1.shape[2:])
                neg2_flat = negatives2.reshape(Btmp * n2, *negatives2.shape[2:])
                negatives = torch.cat([neg1_flat, neg2_flat], dim=0)
                negCounts = torch.full((Btmp,), n1 + n2, dtype=torch.long)
            else:
                raise ValueError(f"Unexpected collate output length: {len(batch)}")

            # 前几轮对每个 query 仅使用一部分负样本 
            if neg_warm_ratio < 0.999:
                # 对每个 query 的 negCount 做上限：ceil(negCount * ratio)，至少 1
                capped = []
                for c in negCounts.tolist():
                    use_c = max(1, int(np.ceil(c * neg_warm_ratio)))
                    capped.append(use_c)
                negCounts_used = torch.tensor(capped, dtype=torch.long)
            else:
                negCounts_used = negCounts

            # ===== 计算用于 loss 的负样本总数，并按顺序截取 =====
            B = int(query.shape[0])
            nNeg_full = int(torch.sum(negCounts).item())
            nNeg_used = int(torch.sum(negCounts_used).item())

            # 拼接输入并到 device
            inp = torch.cat([query, positives, negatives], dim=0).to(device, non_blocking=True).float()
            seq_encoding = model.pool(inp)

            seqQ, seqP, seqN_full = torch.split(seq_encoding, [B, B, nNeg_full], dim=0)

            optimizer.zero_grad()

            # 逐 query/neg 计算损失（仅使用 capped 数量）
            loss = 0.0
            base_full = 0
            base_used = 0
            for i, (c_full, c_used) in enumerate(zip(negCounts.tolist(), negCounts_used.tolist())):
                # 当前 query 的负样本在 seqN_full 中的起止
                cur_full_slice = seqN_full[base_full: base_full + c_full]
                # 只取前 c_used 个参与损失
                cur_used_slice = cur_full_slice[:c_used]
                # 依次加到 loss
                for k in range(c_used):
                    loss = loss + criterion(seqQ[i:i+1], seqP[i:i+1], cur_used_slice[k:k+1])
                base_full += c_full
                base_used += c_used

            # 前几轮缩小 loss，放慢下降
            if loss_warm_ratio < 0.999:
                loss = loss * float(loss_warm_ratio)

            # 归一化（按实际参与的负样本数）
            denom = max(1, nNeg_used)
            loss = loss / float(denom)

            loss.backward()

            # 
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))

            optimizer.step()

            # 清理
            del inp, seq_encoding, seqQ, seqP, seqN_full
            del query, positives, negatives
            if len(batch) == 7:
                del negatives1, negatives2, negCounts1, negCounts2
            else:
                del negatives1, negatives2
            del batch

            batch_loss = float(loss.detach().item())
            epoch_loss += batch_loss

            if iteration % 50 == 0 or nBatches <= 10:
                print(f"==> Epoch[{epoch}]({iteration}/{nBatches}): "
                      f"Loss: {batch_loss:.4f} | "
                      f"loss_warm={loss_warm_ratio:.3f} neg_warm={neg_warm_ratio:.3f} "
                      f"(used {nNeg_used}/{nNeg_full})", flush=True)
                writer.add_scalar('Train/Loss', batch_loss, ((epoch - 1) * nBatches) + iteration)
                writer.add_scalar('Train/nNegUsed', nNeg_used, ((epoch - 1) * nBatches) + iteration)
                writer.add_scalar('Train/nNegFull', nNeg_full, ((epoch - 1) * nBatches) + iteration)
                writer.add_scalar('Train/LossWarmRatio', loss_warm_ratio, ((epoch - 1) * nBatches) + iteration)
                writer.add_scalar('Train/NegWarmRatio', neg_warm_ratio, ((epoch - 1) * nBatches) + iteration)
                if device.type == "cuda":
                    print('Allocated:', torch.cuda.memory_allocated())
                    print('Reserved :', torch.cuda.memory_reserved())

        startIter += len(training_data_loader)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            remove(train_set.cache)
        except FileNotFoundError:
            pass

    avg_loss = epoch_loss / max(1, nBatches)
    print(f"===> Epoch {epoch} Complete: Avg. Loss: {avg_loss:.4f}", flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
