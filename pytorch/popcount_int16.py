import torch


def bitwise_popcount_int16(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the population count (number of set bits) for each element in an int16 tensor.
    Uses only int16 operations, masking after each shift to avoid sign extension issues.

    Args:
        x (torch.Tensor): An int16 tensor.

    Returns:
        torch.Tensor: A tensor of the same shape as x, where each element is the popcount of the corresponding element in x.
    """
    if x.dtype != torch.int16:
        raise TypeError(f"Input tensor must be of dtype torch.int16, got {x.dtype}.")

    # All masks are int16, but we use & 0xFFFF after each shift to avoid sign extension
    m1 = 0x5555
    m2 = 0x3333
    m4 = 0x0F0F
    m8 = 0x00FF
    mask16 = 0xFFFF

    x1 = (x & m1) + (((x >> 1) & mask16) & m1)
    x2 = (x1 & m2) + (((x1 >> 2) & mask16) & m2)
    x3 = (x2 & m4) + (((x2 >> 4) & mask16) & m4)
    x4 = (x3 & m8) + (((x3 >> 8) & mask16) & m8)
    return x4


if __name__ == "__main__":
    # Reference popcount using Python's bin()
    def ref_popcount_int16(val):
        # Mask to 16 bits, treat as unsigned
        return bin(val & 0xFFFF).count('1')

    import torch
    import random

    # Test values: positive, negative, edge cases
    test_vals = [0, 1, -1, 0x7FFF, -0x8000, 0x5555, -0x5556, -21846, -0x5555, 0x1234, -0x1234]
    t = torch.tensor(test_vals, dtype=torch.int16)
    popcounts = bitwise_popcount_int16(t)
    ref = torch.tensor([ref_popcount_int16(v) for v in test_vals], dtype=torch.int16)
    print("Test values:", t.tolist())
    print("bitwise_popcount_int16:", popcounts.tolist())
    print("Reference:", ref.tolist())
    assert torch.equal(popcounts, ref), f"Mismatch: {popcounts.tolist()} vs {ref.tolist()}"
    print("Basic tests passed.")

    # Randomized test
    rand_vals = [random.randint(-32768, 32767) for _ in range(1000)]
    t_rand = torch.tensor(rand_vals, dtype=torch.int16)
    popcounts_rand = bitwise_popcount_int16(t_rand)
    ref_rand = torch.tensor([ref_popcount_int16(v) for v in rand_vals], dtype=torch.int16)
    assert torch.equal(popcounts_rand, ref_rand), "Randomized test failed!"
    print("Randomized tests passed.")

    # Large tensor efficiency test on cpu
    large_tensor = torch.randint(-32768, 32767, (10_000_000,100), dtype=torch.int16)
    import time
    start = time.time()
    popcounts_large = bitwise_popcount_int16(large_tensor)
    elapsed = time.time() - start
    print(f"Popcount on 10M int16 elements on cpu took {elapsed:.4f} seconds.")

    # Large tensor efficiency test on gpu
    large_tensor = large_tensor.to(torch.device("cuda"))
    start = time.time()
    popcounts_large = bitwise_popcount_int16(large_tensor)
    elapsed = time.time() - start
    print(f"Popcount on 10M int16 elements on gpu took {elapsed:.4f} seconds.")