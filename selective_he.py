import torch
from torch.utils.data import DataLoader, Dataset

def get_k_most_sensitivie(map, p):
    top_k_indices = torch.topk(map, int(len(map) * p), largest=True).indices
    Mask = torch.zeros(map.shape) == 0
    Mask[top_k_indices] = False
    return Mask

def calculate_mask(args, global_model, dataset_test):
    Dte = DataLoader(dataset_test, batch_size=args.TB, shuffle=True)
    loss_function = torch.nn.CrossEntropyLoss().to(args.device)
    for batch_idx, (imgs, labels) in enumerate(Dte):
        #print("Enter times: ", batch_idx)
        imgs = imgs.to(args.device)
        labels = labels.type(torch.LongTensor).to(args.device)
        features, y_preds = global_model(imgs)

    loss = loss_function(y_preds, labels)
    loss.backward()
    gradients = [param.grad.data.view(-1) for param in global_model.parameters()]
    dloss_dw = [grad.tolist() for grad in gradients] #flat gradients(tensors) to a vector
    encryption_mask = []

    for grad in gradients:
        # tensor grad
        encryption_mask.append(get_k_most_sensitivie(grad, 0.1))
    #print(encryption_mask)

    return encryption_mask
