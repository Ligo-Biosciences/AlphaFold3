"""This is a naive implementation of the atom attention components. 
 We did early experiments with a PyTorch-native implementation that is supposed to use memory more efficiently, 
 but they did not show much benefit since attention implementations in PyTorch were much slower despite 
 adding considerable clutter and complexity. We fall back to the Deepspeed4Science optimized attention kernel, which reduce 
 the memory consumption to linear anyway. 

However, this is not recommended for large scale training. 
The smart move here will be to migrate to FlexAttention once there is bias gradient support.
"""
