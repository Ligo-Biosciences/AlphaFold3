"""The Pairformer stack fulfills a similar role to the Evoformer stack in AlphaFold2, the Pairformer stack uses just
a single representation rather than a representation for a subset of the MSA. The single representation does not
influence the pair representation, the pair representation is used to control information flow in the single
representation by biasing the Attention logits.
All transition blocks use SwiGlu.
"""