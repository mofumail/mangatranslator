from transformers import PretrainedConfig

class HayaiConfig(PretrainedConfig):
    model_type = "hayai"
    def __init__(self, vocab_size=0, d_model=512, d_vision=768, d_ffn=2048, n_layers=12, **kwargs):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_vision = d_vision
        self.d_ffn = d_ffn
        self.n_layers = n_layers
        super().__init__(**kwargs)
