from torch import nn

class AuraParallelAdapterMLP(nn.Module):
    def __init__(self, config, adapter_dim, adapter_scaling):
        super().__init__()
        self.config = config
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.adapter_down = nn.Linear(self.hidden_size, adapter_dim, bias=False)
        self.adapter_up = nn.Linear(adapter_dim, self.hidden_size, bias=False)
        self.adapter_act = nn.GELU()

        self.adapter_dropout = nn.Dropout(p=config.adapter_dropout)
        self.adapter_scaling = adapter_scaling

    def forward(self, x):
        x = self.adapter_dropout(x)
        x = self.adapter_scaling * self.adapter_up(
            self.adapter_act(self.adapter_down(x))
        )
        return x
