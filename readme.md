implementation of https://arxiv.org/abs/1605.08803, "Density Estimation using Real NVP"

Gemini2.5 told me it was the OG normalizing flows paper 

~10 hours from reading paper to getting something basic to run 

divergences with paper:
- No batchnorm (equinox batchnorm is stateful)
- No residual connections
The hidden layers im using are pretty small 

Sometimes value for s() explodes, giving nans, should try scaling input by 0.01 maybe
