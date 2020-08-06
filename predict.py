import torch
from tqdm import tqdm

def fastPredictionGpu(
                    embed_user,
                    embed_item,
                    bias_user,
                    bias_item,
                    rate_mean,
                    filter_list = None,
                    batch = 512,
                    topk = 10):
    """
    predict all negative results for user of topk 
    embed_user: user embedding
    embed_item: item embedding
    bias_user: user bias
    bias_item: user bias
    filter_list: filter list for user used item
    """
    results_index = None
    results_score = None
    for i in tqdm(range(0, embed_user.shape[0], batch)):
        mask = embed_user.new_ones([min(batch, embed_user.shape[0] - i), embed_item.shape[0]])
        for j in range(batch):
            if i + j >= embed_user.shape[0]:
                break
            # mask[j].scattrer_(dim = 0, index = torch.tensor(list(filter_list[i + j])).cuda(), value = torch.tensor(0.0).cuda())
        # calculate distance between user embedding and item embedding for current batch
        pred = torch.mm(embed_user[i: i + min(batch, embed_user.shape[0] - i), :], embed_item.t()) + bias_user[i: i + min(batch, embed_user.shape[0] - i)] + bias_item.t() + rate_mean
        # filter used item
        # pred = torch.mul(mask, pred)
        # get TopK
        topk_score, topk_index = torch.topk(pred, k = topk, dim = 1)
        # concat results
        if results_index is not None:
            results_index = torch.cat((topk_index, results_index), dim = 0)
        else:
            results_index = topk_index

        if results_score is not None:
            results_score = torch.cat((topk_score, results_score), dim = 0)
        else:
            results_score = topk_score

    return results_score, results_index
