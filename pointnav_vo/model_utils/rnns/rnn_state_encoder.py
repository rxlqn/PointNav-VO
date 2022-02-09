import torch
import torch.nn as nn
import numpy as np

class RNNStateEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "GRU",
    ):
        r"""An RNN for encoding the state in RL.

        Supports masking the hidden state during various timesteps in the forward lass

        Args:
            input_size: The input size of the RNN
            hidden_size: The hidden size
            num_layers: The number of recurrent layers
            rnn_type: The RNN cell type.  Must be GRU or LSTM
        """

        super().__init__()
        self._num_recurrent_layers = num_layers
        self._rnn_type = rnn_type

        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
        )

        self.layer_init()

    def layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    ## 不知道这里为什么LSTM要乘2
    @property
    def num_recurrent_layers(self):
        return self._num_recurrent_layers * (2 if "LSTM" in self._rnn_type else 1)

    def _pack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = torch.cat([hidden_states[0], hidden_states[1]], dim=0)

        return hidden_states

    def _unpack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = (
                hidden_states[0 : self._num_recurrent_layers],
                hidden_states[self._num_recurrent_layers :],
            )

        return hidden_states

    def _mask_hidden(self, hidden_states, masks):
        """This function ensures that every episode starts with zero-valued hidden states.
        """
        if isinstance(hidden_states, tuple):
            hidden_states = tuple(v * masks for v in hidden_states)
        else:
            hidden_states = masks * hidden_states

        return hidden_states

    def single_forward(self, x, hidden_states, masks):
        r"""Forward for a non-sequence input
        """
        hidden_states = self._unpack_hidden(hidden_states)
        x, hidden_states = self.rnn(
            x.unsqueeze(0), self._mask_hidden(hidden_states, masks.unsqueeze(0)),
        )
        x = x.squeeze(0)
        hidden_states = self._pack_hidden(hidden_states)
        return x, hidden_states, 0

    def seq_forward(self, x, hidden_states, masks):
        r"""Forward for a sequence of length T

        Args:
            x: (T, N, -1) Tensor that has been flattened to (T * N, -1)
            hidden_states: The starting hidden state.
            masks: The masks to be applied to hidden state at every timestep.
                A (T, N) tensor flatten to (T * N)
            x: 128*1*-1拆成32*4*-1  截断GRU
            传入的hidden_states改成了整个episode的hidden_states 128 2 1 512
        """
        # x is a (T, N, -1) tensor flattened to (T * N, -1)
        n = hidden_states.size(2)
        t = int(x.size(0) / n)

        # unflatten
        x = x.view(t, n, x.size(1)).contiguous()
        masks = masks.view(t, n).contiguous()

        # steps in sequence which have zero for any agent. Assume t=0 has
        # a zero in it.
        ## mask中是0的下标
        has_zeros = (
            (masks[1:] == 0.0).any(dim=-1).nonzero(as_tuple=False).squeeze().cpu()
        )

        # +1 to correct the masks[1:]
        if has_zeros.dim() == 0:
            has_zeros = [has_zeros.item() + 1]  # handle scalar
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [t]

        hidden_states = self._unpack_hidden(hidden_states)
        outputs = []
        index = []      ## 存新生成的序列
        ## 把非0区间分段处理 0是done 1是不done
        ## 如果小于32怎么办，小于32就按照小的训练，大于32就random sample其中的32部分
        for i in range(len(has_zeros) - 1):
            # process steps that don't have any zeros in masks together
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]
            seq_len = end_idx - start_idx
            sel_len = 16   ## 采样长度        
            if seq_len > sel_len:    ## 如果大于32就要random sample成batch, 否则正常训练
                batch_size = 2
                sample_index_list = [np.random.randint(0, seq_len-sel_len+1) + start_idx for i in range(batch_size)]
                # h_tmp      = self.sample_batch_seq(batch_size, hidden_states, sample_index_list, sel_len)
                # m_tmp      = self.sample_batch_seq(batch_size, masks.view(-1, 1, 1), sample_index_list, sel_len)
                x_tmp      = self.sample_batch_seq(batch_size, x, sample_index_list, sel_len)
                rnn_scores, tmp_hidden_states = self.rnn(
                    x_tmp,
                    self._mask_hidden(
                        torch.cat([hidden_states[i] for i in sample_index_list], 1), 
                        torch.stack([masks[i] for i in sample_index_list], 0).view(1, -1, 1).contiguous()
                    ),
                )
                rnn_scores = rnn_scores.reshape(-1, 1, rnn_scores.size(2)) ## T N -1 -> T*N 1 -1
                for i in sample_index_list:
                    index.append(torch.linspace(i, i+sel_len-1, sel_len))

            else:   ## 小于32就直接推理 这个batch和上面的情况不匹配
                rnn_scores, tmp_hidden_states = self.rnn(
                    x[start_idx:end_idx],
                    self._mask_hidden(
                        hidden_states[start_idx], masks[start_idx].view(1, -1, 1).contiguous()
                    ),
                )
                index.append(torch.linspace(start_idx, end_idx-1, end_idx-start_idx)) 
            outputs.append(rnn_scores)

        # x is a (T, N, -1) tensor x的大小不一定是T*N
        x = torch.cat(outputs, dim=0)
        x = x.view(-1, x.size(2)).contiguous()  # flatten

        hidden_states = self._pack_hidden(hidden_states)
        return x, hidden_states, torch.cat(index, dim=0).long()

    def forward(self, x, hidden_states, masks):
        if x.size(0) == hidden_states.size(1):
            return self.single_forward(x, hidden_states, masks)
        else:
            return self.seq_forward(x, hidden_states, masks)

    def sample_batch_seq(self, batch_size, seq, sample_index_list, sel_len):
        '''
        input: T N D
        '''
        seq_tmp = []
        for i in range(batch_size):
            index = sample_index_list[i]    ## 0-250    ## 随机sample固定seq_len长度
            seq_tmp.append(seq[index:index+sel_len:,])
        if len(seq.size()) == 4:            ## hidden_states
            return torch.cat(seq_tmp, dim=2)
        else:
            return torch.cat(seq_tmp, dim=1)