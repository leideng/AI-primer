import torch
import torch.nn.functional as F
# Example of using softmax in PyTorch
# This will compute the softmax along dimension 1 (columns)
# and return a tensor with the same shape as the input tensor.
# The output will be a probability distribution over the columns.
# The sum of each row will be 1.
# The custom implementation of softmax is also provided for verification.
# The custom implementation uses the log-sum-exp trick for numerical stability.
# It subtracts the maximum value in each row before exponentiating,
# and then normalizes by the sum of the exponentials.
# This helps prevent overflow and underflow issues in the exponentiation step.
# The custom implementation should yield the same result as the built-in softmax function.
# The output of the custom implementation is compared with the built-in softmax function
# to ensure correctness and numerical stability.    
# The custom implementation is verified by checking if the outputs are close enough
# and if the sum of their absolute differences is within a small tolerance.
# The output of the custom implementation is expected to be very close to the built-in softmax
# function, with a very small difference due to numerical precision.
# Example tensor
#list_data = [[1, 2, -1, -2], [3, 4, -4, -3]]
list_data = [[100, 200, -100, -200], [300, 400, -400, -300]]
#x = torch.randn(2, 4)
x = torch.tensor(list_data, dtype=torch.float32)
print(f"x:\n{x}")
output = F.softmax(x, dim=1)
#print(f"Softmax output: {output}")
print(f"Softmax output:\n{torch.round(output * 1000) / 1000}")

# Custom implementation of softmax using the log-sum-exp trick
maxes = torch.max(x, 1, keepdim=True)[0]
x_exp = torch.exp(x-maxes)
x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
output_custom = x_exp/x_exp_sum
print(f"Customized softmax output:\n{torch.round(output_custom * 1000) / 1000}")

# Verify that the custom implementation matches the built-in softmax
# This will print True if they are close enough, and the sum of absolute differences
print(torch.allclose(output, output_custom))
print(torch.sum(torch.abs(output-output_custom)))
