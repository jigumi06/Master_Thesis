import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset

import utils.optimizer as op
import tenseal as ts
import utils.ckks as ckks

#class client:
def client_update(args, client_index, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} training...'.format(k))
        client_models[k], loss = op.fedavg_optimizer(args, client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
        
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        #print(index_nonselect)
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
    return client_models, loss_dict

def client_update_iid_feature(args, client_index, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} training...'.format(k))
        client_models[k], loss = op.fedavg_optimizer(args, client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
        
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        #print(index_nonselect)
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
    return client_models, loss_dict



def client_prox_update(args, client_index, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} training...'.format(k))
        client_models[k], loss = op.fedprox_optimizer(args, client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
        
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        #print(index_nonselect)
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
    return client_models, loss_dict

def client_LC_update(args, client_index, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} training...'.format(k))
        client_models[k], loss = op.fedLC_optimizer(args, client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
        
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        #print(index_nonselect)
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
    return client_models, loss_dict


def client_feddyn(args, client_index, pre_client_models, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    n_clnt = args.K
    weight_list = np.asarray([len(dict_users[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} training...'.format(k))

        client_models[k],  loss = op.feddyn_optimizer(args, pre_client_models[k], client_models[k], global_model, global_round, dataset_train, dict_users[k],weight_list[k]) 
        loss_dict[k].extend(loss)
        
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        #print(index_nonselect)
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
    return client_models, loss_dict



def client_moon(args, client_index, preround_client_models, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_moon...'.format(k))

        client_models[k], loss = op.moon_optimizer(args, preround_client_models[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               

    return client_models, loss_dict

def client_fedproc(args, client_index, contrastiveloss_funcs, client_models, global_model, global_round, 
                   dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index:#k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        contrastiveloss_funcs[k], client_models[k], loss = op.fedproc_optimizer(args, contrastiveloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return contrastiveloss_funcs, client_models, loss_dict


def calculate_sensitivity_map(args, global_model, dataset, dataset_index):
    sensitivity_maps = []
    
    for k in range(args.K):
        Dtr = DataLoader(op.DatasetSplit(dataset, dataset_index[k]), batch_size=args.B, shuffle=True)
        loss_function = torch.nn.CrossEntropyLoss().to(args.device)
        for batch_idx, (imgs, labels) in enumerate(Dtr):
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, y_preds = global_model(imgs)

        loss = loss_function(y_preds, labels)
        loss.backward()
        gradients = [torch.abs(param.grad.data) for param in global_model.parameters()]
        sensitivity_maps.append(gradients)

    return sensitivity_maps

def decryption_to_plain(layer):
    ckks_count = 0
    if isinstance(layer, list):
        return [decryption_to_plain(sublayer) for sublayer in layer]
    elif isinstance(layer, ts.CKKSVector):
        ckks_count += 1
        # print("CKKSVector")
        return layer.decrypt()[0]
    else:
        # print("count of CKKSVector: ", ckks_count)
        return layer

def client_fedfa_cl(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_cl_optimizer(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[args.K-1])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict

def client_fedfa_cl_secured(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict, masks, layer_list, global_list, shape_list, protected):  # update nn
    #client_models = [[] for i in range(args.K)]
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))
        
        if global_round == 0:
            client_models[k] = copy.deepcopy(global_model)

        # if global_round != 0:
        #     client_model = decryption_of_client_model(args, layer_list[k], global_model)

        # Decrypt by parameters
        '''
        if global_round != 0:
            client_model = decrypt_by_para(args, global_round, layer_list[k], shape_list, global_model)
        anchorloss_funcs[k], client_model, loss = op.fedfa_cl_optimizer(args, anchorloss_funcs[k], client_model, global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)

        # Encryption on parameters
        encrypt_by_para(args, layer_list[k], client_model, masks)
        '''

        if global_round != 0:
            #client_model = decryption_of_client_model(args, layer_list[k], global_model)
            # Decrypt by parameters
            client_models[k] = decrypt_by_para(args, global_round, layer_list, shape_list, k, global_model)
        
        if protected:
            dataset_index = k
        else:
            #dataset_index = int(args.K/2) + k
            dataset_index = args.K-1
        print("dataset_index: ", dataset_index)
        anchorloss_funcs[k], client_models[k], loss = op.fedfa_cl_optimizer(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[dataset_index])
        
        loss_dict[k].extend(loss)
        index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        for j in index_nonselect:
            loss = [loss_dict[j][-1]]*args.E 
            loss_dict[j].extend(loss)  
    
        if protected:
            # Encryption on parameters
            temp = encrypt_by_para(args, layer_list[k], client_models[k], k, masks)
            layer_list[k] = temp
            
    return anchorloss_funcs, client_models, loss_dict, layer_list
            

def decryption_of_client_model(args, client_model_list, model):
    # temp = copy.deepcopy(client_model_list)
    # for key in temp:
    #     enc_weight = temp[key]
    #     if key != 'fc1.weight':
    #         temp[key] = torch.Tensor(enc_weight.decrypt().tolist())
    #     else:
    #         temp[key] = enc_weight
    # model.load_state_dict(temp)

    # return model

    for key in client_model_list:
        enc_weight = client_model_list[key]
        if key != 'fc1.weight':
            client_model_list[key] = torch.Tensor(enc_weight.decrypt().tolist())
        else:
            client_model_list[key] = enc_weight
    model.load_state_dict(client_model_list)

    return model

# skip fc1.weight
def encryption_of_client_model(args, client_model, masks):
    # model_dict = copy.deepcopy(client_model.state_dict())
    print("start encryption")
    model_dict = client_model.state_dict()
    for key, mask_layer in zip(model_dict, masks):
        if key != "fc1.weight":
            layer = model_dict[key]
            mask_model_layer = torch.mul(mask_layer, layer)
            encrypted_mask_model = ckks.EncryptionManager().encrypt_layer(mask_model_layer)
            print(" type of encrypted_mask_model",type(encrypted_mask_model))
            bool_mask_layer = mask_layer == 0
            plain_layer = torch.mul(bool_mask_layer, layer)
            # .cpu().detach().numpy()
            print(" type of plain_layer",type(plain_layer))

            layer = encrypted_mask_model + plain_layer
            print(" type of layer", type(layer))

            model_dict[key] = layer
    print("stop encryption")
    return model_dict


def decrypt_by_para_deleted(args, global_round, layer_list, shape_list, model):
    print("Enter <decrypt_by_para>")
    print("layer_list: ", layer_list)
    client_model = copy.deepcopy(model)
    if global_round != 0 :
        decrypted_layers = []
        layer_count = 0
        for layer_key, layer_encrypted in layer_list.items():
            layer_count += 1
            if (layer_count != 3):
                decrypted_layer = decryption_to_plain(layer_encrypted)
            else:
                decrypted_layer = layer_encrypted
            decrypted_layers.append(decrypted_layer)
        #print("decrypted_layers: ", decrypted_layers)        
        reshaped_tensors = []
        for layer_data, shape in zip(decrypted_layers, shape_list):
            tensor = torch.Tensor(layer_data).detach().view(shape).to(args.device)
            reshaped_tensors.append(tensor)

    
        with torch.no_grad():  # Ensure we do not track these operations in the gradient computation
            for param, new_data in zip(client_model.parameters(), decrypted_layers):
                param.data.copy_(torch.Tensor(new_data.tolist()))

        print("End <decrypt_by_para>")
        return client_model
    

'''
def encrypt_by_para(args, layer_list, client_model, masks):
    print("Enter <encrypt_by_para>")
    layer_list = []
    layer_count = 0
    for layer, mask_layer in zip(client_model.parameters(), masks):
        layer_count += 1
        # print("layer size: ", layer.size())
        if (layer_count != 3):
            # layer_shape = layer.shape
            flat_tensor = layer.flatten()
            flat_tensor_list = flat_tensor.tolist()
            # print(" layer size: ", len(flat_tensor_list))
            flat_mask = mask_layer.flatten()
            flat_mask_list = flat_mask.tolist()
            # print(" mask size: ", len(flat_mask_list))

            layer = [ ckks.EncryptionManager().encrypt_vector(flat_tensor_list[i]) 
                            if flat_mask_list[i] == 1 else flat_tensor_list[i] for i in range(len(flat_mask_list))]
        
        layer_list.append(layer)
    print("End <encrypt_by_para>")
    return layer_list
'''

def decrypt_by_para(args, global_round, layer_list, shape_list, k, model):
    print("<Enter decrypt>")
    if global_round != 0 :
    
        decrypted_layers = []
        layer_count = 0
        for layer_encrypted in layer_list[k]:
            layer_count += 1
            if (layer_count != 3):
                decrypted_layer = decryption_to_plain(layer_encrypted)
            else:
                decrypted_layer = layer_encrypted
            #decrypted_layer = decryption_to_plain(layer_encrypted)
            #decrypted_layers.append(decrypted_layer)
        
        reshaped_tensors = []
        for layer_data, shape in zip(decrypted_layers, shape_list):
            tensor = torch.Tensor(layer_data).detach().view(shape).to(args.device)
            reshaped_tensors.append(tensor)

    
        with torch.no_grad():  # Ensure we do not track these operations in the gradient computation
            for param, new_data in zip(model.parameters(), decrypted_layers):
                param.data.copy_(torch.Tensor(new_data.tolist()))

        # print("stop decryption")
        print("<Finish decrypt>")
        return model


def encrypt_by_para(args, layer_list, client_model, k, masks):
    print("<Enter encrypt>")
    layer_count = 0
    temp = []
    for layer, mask_layer in zip(layer_list, masks):
        layer_count += 1
        # print("layer size: ", layer.size())
        if (layer_count != 3):
            # layer_shape = layer.shape
            flat_tensor = layer.flatten()
            flat_tensor_list = flat_tensor.tolist()
            # print(" layer size: ", len(flat_tensor_list))
            flat_mask = mask_layer.flatten()
            flat_mask_list = flat_mask.tolist()
            # print(" mask size: ", len(flat_mask_list))
    
            layer = [ ckks.EncryptionManager().encrypt_vector(flat_tensor_list[i]) 
                                if flat_mask_list[i] == 1 else flat_tensor_list[i] for i in range(len(flat_mask_list))]
        
        temp.append(layer)
    print("<Finish encrypt>")
    return temp

######################ablation####################### >>>>>> fedfa_without_anchor_updating
def client_fedfa_cl_without_anchor_updating(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_without_anchor_updating(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict

######################ablation####################### >>>>>> fedfa_without_classifer_calibration
def client_fedfa_cl_without_classifer_calibration(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_without_classifer_calibration(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict

######################ablation####################### >>>>>> fedfa_with_epoch_classifer_calibration
def client_fedfa_cl_with_post_classifer_calibration(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_with_post_classifer_calibration(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict

######################ablation####################### >>>>>> fedfa_with_epoch_classifer_calibration
def client_fedfa_cl_with_epoch_classifer_calibration(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_with_epoch_classifer_calibration(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict

######################ablation####################### >>>>>> fedfa_with_epoch_classifer_calibration
def client_fedfa_cl_with_pre_classifer_calibration(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_with_pre_classifer_calibration(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict

######################ablation####################### >>>>>> fedfa_without_anchor_specfic_initialization
def client_fedfa_cl_without_anchor_specfic_initialization(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_without_anchor_specfic_initialization(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict


######################ablation####################### >>>>>> fedfa_with_anchor_oneround_initialization
def client_fedfa_cl_with_anchor_oneround_initialization(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_with_anchor_oneround_initialization(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict

