# predictor.py

from torch.nn.functional import pad
from Predictor.predictor_models import *

import torch
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class PREDICT:
    def __init__(self, model_path):
        self.input_size = 3
        self.hidden_size = 256
        self.output_size = 1
        self.lambda_l2 = 0.001
        self.dropout_rate = 0.2
        self.model_path = model_path
        self.use_gpu = True if torch.cuda.is_available() else False
        self.model = self.load_model()

    def load_model(self):
        model = LSTMModel(self.input_size, self.hidden_size, self.output_size, self.dropout_rate, self.lambda_l2)
        model.load_state_dict(torch.load(self.model_path))
        if self.use_gpu:
            model.cuda()
        return model

    def string_to_array(self, my_string):
        my_string = my_string.lower()
        my_string = re.sub('[^acgt]', 'z', my_string)
        my_array = np.array(list(my_string))
        return my_array

    def one_hot_encode(self, my_array):
        label_encoder = LabelEncoder()
        label_encoder.fit(np.array(['a', 'c', 'g', 't', 'z']))
        integer_encoded = label_encoder.transform(my_array)
        onehot_encoder = OneHotEncoder(sparse=False, dtype=int)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        onehot_encoded = np.delete(onehot_encoded, -1, 1)
        return onehot_encoded

    def seq_onehot(self, seq):
        onehot_seq = [torch.tensor(self.one_hot_encode(self.string_to_array(s)), dtype=torch.float32) for s in seq]

        max_length = max(matrix.shape[0] for matrix in onehot_seq)

        padded_tensor_list = []
        for matrix in onehot_seq:
            padding_length = max_length - matrix.shape[0]
            padded_tensor = pad(matrix, (0, 0, 0, padding_length), value=0)
            padded_tensor_list.append(padded_tensor)

        onehot_seq = torch.stack(padded_tensor_list, dim=0)

        return onehot_seq

    def pre_seqs(self, population):
        """
            Batch prediction of sequence fitness.

            Parameters:
            - population: [{'sequence': sequence, 'expression': None}, ...]

            Return value:
            - population: The population with updated expression values
        """
        input_seqs = [ind['sequence'] for ind in population]

        valseq_onehot = self.seq_onehot(input_seqs)
        valseq = torch.Tensor(valseq_onehot).cuda() if self.use_gpu else torch.Tensor(valseq_onehot)

        with torch.no_grad():
            val_output = self.model(valseq)

        val_output = val_output.cpu().numpy()
        val_pred = val_output.flatten().tolist()
        for i, ind in enumerate(population):
            ind['expression'] = val_pred[i]

        return population
