import torch
from attention import SelfAttention, CausalAttention, MultiHeadAttention


inputs = torch.tensor(
   [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = inputs.shape[-1]
d_out = 2

torch.manual_seed(0)
sa = SelfAttention(d_in, d_out)

print(sa(inputs))

batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1] #num_tokens
ca = CausalAttention(d_in, d_out, context_length, 0.1)
context_vec = ca(batch)
print(context_vec)

batch_size, context_length, d_in = batch.shape
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

print(mha(batch))
