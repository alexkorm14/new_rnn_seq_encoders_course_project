# эта реализация не работает - https://github.com/muditbhargava66/PyxLSTM/tree/main/xLSTM

import torch
import torch.nn as nn
import torch.nn.functional as F

class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstms = nn.ModuleList([nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        self.exp_forget_gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.exp_input_gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        
        self.reset_parameters()

    def reset_parameters(self):
        for lstm in self.lstms:
            nn.init.xavier_uniform_(lstm.weight_ih)
            nn.init.xavier_uniform_(lstm.weight_hh)
            nn.init.zeros_(lstm.bias_ih)
            nn.init.zeros_(lstm.bias_hh)
        
        for gate in self.exp_forget_gates + self.exp_input_gates:
            nn.init.xavier_uniform_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(self, input_seq, hidden_state=None):
        batch_size = input_seq.size(0)
        seq_length = input_seq.size(1)

        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)

        output_seq = []
        for t in range(seq_length):
            x = input_seq[:, t, :]
            new_hidden_state = []
            for idx, (lstm, dropout, f_gate, i_gate) in enumerate(list(zip(self.lstms, self.dropout_layers, self.exp_forget_gates, self.exp_input_gates))):
                if hidden_state[idx][0] is None:
                    h, c = lstm(x)
                else:
                    h, c = lstm(x, (hidden_state[idx][0], hidden_state[idx][1]))

                f = torch.exp(f_gate(h))
                i = torch.exp(i_gate(h))
                c = f * c + i * lstm.weight_hh.new_zeros(batch_size, self.hidden_size)

                new_hidden_state.append((h, c))

                if idx < self.num_layers - 1:
                    x = dropout(h)
                else:
                    x = h

            hidden_state = new_hidden_state
            output_seq.append(x)

        output_seq = torch.stack(output_seq, dim=1)
        return output_seq, hidden_state

    def init_hidden(self, batch_size):
        hidden_state = []
        for lstm in self.lstms:
            h = torch.zeros(batch_size, self.hidden_size, device=lstm.weight_ih.device)
            c = torch.zeros(batch_size, self.hidden_size, device=lstm.weight_ih.device)
            hidden_state.append((h, c))
        return hidden_state

class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstms = nn.ModuleList([nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)

        self.exp_input_gates = nn.ModuleList([nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.exp_forget_gates = nn.ModuleList([nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.output_gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        
        self.reset_parameters()

    def reset_parameters(self):
        for lstm in self.lstms:
            nn.init.xavier_uniform_(lstm.weight_ih)
            nn.init.xavier_uniform_(lstm.weight_hh)
            nn.init.zeros_(lstm.bias_ih) 
            nn.init.zeros_(lstm.bias_hh)
        
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias) 
        nn.init.zeros_(self.W_v.bias)
        
        for gate in self.exp_input_gates + self.exp_forget_gates + self.output_gates:
            nn.init.xavier_uniform_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(self, input_seq, hidden_state=None):
        batch_size = input_seq.size(0)
        seq_length = input_seq.size(1)

        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)

        output_seq = []
        for t in range(seq_length):
            x = input_seq[:, t, :]
            queries = self.W_q(x)
            keys = self.W_k(x)
            values = self.W_v(x)

            new_hidden_state = []
            for idx, (lstm, dropout, i_gate, f_gate, o_gate) in enumerate(list(zip(self.lstms, self.dropout_layers, self.exp_input_gates, self.exp_forget_gates, self.output_gates))):
                if hidden_state[idx][0] is None:
                    h, C = lstm(x)
                else:
                    h, C = hidden_state[idx]
                
                i = torch.exp(i_gate(x))
                f = torch.exp(f_gate(x))
                o = nn.functional.sigmoid(o_gate(h))
            
                C_t = f * C + torch.matmul(torch.matmul(values, keys.T),i)
                print(torch.matmul(torch.matmul(values, keys.T),i).isnan().sum())
                print(C_t)
                print(C_t.isnan().sum())
                print()
                attn_output = C_t * queries
                
                h = o * attn_output
                new_hidden_state.append((h, C_t))
                
                if idx < self.num_layers - 1:
                    x = dropout(h)
                else:
                    x = h
            
            hidden_state = new_hidden_state
            output_seq.append(x)

        output_seq = torch.stack(output_seq, dim=1)
        return output_seq, hidden_state

    def init_hidden(self, batch_size):
        hidden_state = []
        for lstm in self.lstms:
            h = torch.zeros(batch_size, self.hidden_size, device=lstm.weight_ih.device)
            C = torch.zeros(batch_size, self.hidden_size, device=lstm.weight_ih.device)
            hidden_state.append((h, C))
        return hidden_state

class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, bidirectional=False, lstm_type="slstm"):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        assert bidirectional != True, 'bidirectional is not supported'
        self.bidirectional = bidirectional
        self.lstm_type = lstm_type

        if lstm_type == "slstm":
            self.lstm = sLSTM(input_size, hidden_size, num_layers, dropout)
        elif lstm_type == "mlstm":
            self.lstm = mLSTM(input_size, hidden_size, num_layers, dropout)
        else:
            raise ValueError(f"Invalid LSTM type: {lstm_type}")

        self.norm = nn.LayerNorm(input_size)
        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)

        if bidirectional:
            self.proj = nn.Linear(2 * hidden_size, input_size)
        else:
            if lstm_type == "mlstm":
                self.up_proj = nn.Sequential(
                    nn.Linear(input_size, 4 * input_size), 
                    nn.GELU(),
                    nn.Linear(4 * input_size, input_size)
                )
            self.proj = nn.Linear(hidden_size, input_size)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "up_proj"):
            nn.init.xavier_uniform_(self.up_proj[0].weight)
            nn.init.zeros_(self.up_proj[0].bias)
            nn.init.xavier_uniform_(self.up_proj[2].weight)
            nn.init.zeros_(self.up_proj[2].bias)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, input_seq, hidden_state=None):
        if hasattr(self, "up_proj"):
            input_seq = self.up_proj(input_seq)

        lstm_output, hidden_state = self.lstm(input_seq, hidden_state)
        if self.lstm_type == "slstm":
            hidden_state = [[hidden_state[i][0], hidden_state[i][1]] for i in range(len(hidden_state))]

        if self.bidirectional:
            lstm_output = torch.cat((lstm_output[:, :, :self.hidden_size], lstm_output[:, :, self.hidden_size:]), dim=-1)

        output = self.activation(self.proj(lstm_output))
        output = self.norm(output + input_seq)
        output = self.dropout_layer(output)

        return output, hidden_state