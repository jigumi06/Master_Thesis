
import tenseal as ts

class EncryptionManager:
    def __init__(self, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60]):
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree, coeff_mod_bit_sizes)
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40
        # self.nns = [[] for i in range(self.args.K)]

    def is_sensitive_parameter(self, key):
        return "bias" in key

    def encrypt_layer(self, layer):
        return ts.ckks_tensor(self.context, layer.cpu().detach().numpy())

    def encrypt(self, dic):
        model_dic = dic.state_dict()

        for key in model_dic:
            if self.is_sensitive_parameter(key):
                tensor_np = model_dic[key].cpu().detach().numpy()
                model_dic[key] = ts.ckks_tensor(self.context, tensor_np)

        return dic

    def decrypt_client(self, dic):
        
        model_dic = dic.state_dict()
        for key in model_dic:
            if self.is_sensitive_parameter(key):
                model_dic[key] = model_dic[key].decrypt()

        return dic


