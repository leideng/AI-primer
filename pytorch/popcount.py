import torch


def bitwise_popcount(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the population count (number of set bits) for each element
    in an integer tensor.

    Args:
        x (torch.Tensor): An integer tensor. Supported dtypes: int8, int16,
                          int32, int64, uint8
                          (Note: PyTorch only natively supports uint8,
                          for others you might need to convert or ensure
                          they are treated as unsigned for correct popcount. So we do not consider uint16, uint32, uint64).

    Returns:
        torch.Tensor: A tensor of the same shape as x, where each element
                      is the popcount of the corresponding element in x.
    """
    # Check if input tensor is of integer type
    integer_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
    if x.dtype not in integer_dtypes:
        raise TypeError(f"Input tensor must be of integral type, got {x.dtype}.")

    # Ensure the tensor is on CPU for intermediate operations if needed,
    # or handle specific CUDA limitations for certain integer types.
    # For common int32/int64, bitwise ops work fine on CUDA.

    # Convert to int64 to handle all common integer sizes up to 64-bit
    # and ensure consistent behavior for bitwise ops.
    original_dtype = x.dtype
    x_64 = x.long() # Converts to int64

    # Mask to keep only the relevant bits for smaller integer types
    if original_dtype == torch.int8 or original_dtype == torch.uint8:
        x_64 = x_64 & 0xFF
    elif original_dtype == torch.int16:
        x_64 = x_64 & 0xFFFF
    elif original_dtype == torch.int32:
        x_64 = x_64 & 0xFFFFFFFF
    # int64: no mask needed

    # The following is a common bit-parallel popcount algorithm.
    # It works by summing adjacent bit counts.
    # Example for 32-bit integers, extendable to 64-bit.
    # For 64-bit, you'd extend the masks and shifts.

    # Masks for 64-bit integers
    m1 = 0x5555555555555555  # 01010101...
    m2 = 0x3333333333333333  # 00110011...
    m4 = 0x0F0F0F0F0F0F0F0F  # 00001111...
    m8 = 0x00FF00FF00FF00FF  # 0000000011111111...
    m16 = 0x0000FFFF0000FFFF # 00000000000000001111111111111111...
    m32 = 0x00000000FFFFFFFF # 0000000000000000000000000000000011111111111111111111111111111111...

    x_64 = (x_64 & m1) + ((x_64 >> 1) & m1)
    x_64 = (x_64 & m2) + ((x_64 >> 2) & m2)
    x_64 = (x_64 & m4) + ((x_64 >> 4) & m4)
    x_64 = (x_64 & m8) + ((x_64 >> 8) & m8)
    x_64 = (x_64 & m16) + ((x_64 >> 16) & m16)
    x_64 = (x_64 & m32) + ((x_64 >> 32) & m32) # For 64-bit integers

    return x_64.to(original_dtype) # Convert back to original dtype if desired

# Example Usage:
if __name__ == "__main__":
    import time
    # Example 1: Positive integers
    a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 15, 16], dtype=torch.int32)
    # Binary:
    # 1: 0001 (1 set bit)
    # 2: 0010 (1 set bit)
    # 3: 0011 (2 set bits)
    # 4: 0100 (1 set bit)
    # 5: 0101 (2 set bits)
    # 6: 0110 (2 set bits)
    # 7: 0111 (3 set bits)
    # 8: 1000 (1 set bit)
    # 15: 1111 (4 set bits)
    # 16: 10000 (1 set bit)
    popcounts_a = bitwise_popcount(a)
    print(f"Tensor a: {a}")
    print(f"Popcounts for a: {popcounts_a}")
    # Expected: [1, 1, 2, 1, 2, 2, 3, 1, 4, 1]

    print("-" * 30)

    # Example 2: Larger numbers, including a 64-bit example
    b = torch.tensor([0b10101010101010101010101010101010, # 32-bit, 16 set bits
                      -1, # All 64 bits set (two's complement)
                      0], dtype=torch.int64)
    popcounts_b = bitwise_popcount(b)
    print(f"Tensor b: {b}")
    print(f"Popcounts for b: {popcounts_b}")
    # Expected: [16, 64, 0]

    print("-" * 30)

    # Example 3: On CUDA (if available)
    if torch.cuda.is_available():
        c = torch.randint(0, 2**31 - 1, (2, 3), dtype=torch.int32, device='cuda')
        popcounts_c = bitwise_popcount(c)
        print(f"Tensor c (on CUDA): {c}")
        print(f"Popcounts for c (on CUDA): {popcounts_c}")
    else:
        print("CUDA not available, skipping CUDA example.")

    print("-" * 30)

    # Example 4: Using uint8 (which is natively supported by PyTorch)
    d = torch.tensor([0b11110000, 0b00001111, 0b10101010], dtype=torch.uint8)
    popcounts_d = bitwise_popcount(d)
    print(f"Tensor d (uint8): {d}")
    print(f"Popcounts for d (uint8): {popcounts_d}")
    # Expected: [4, 4, 4]

    # --- Popcount tests for all dtypes ---
    print("Testing all dtypes...")
    dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
    test_cases = {
        torch.int8: [([0, -1], [0, 8]), ([85, -86], [4, 4])],  # 85 = 0b01010101, -86 = 0b10101010
        torch.int16: [([0, -1], [0, 16]), ([21845, -21846], [8, 8])],  # 21845 = 0b0101010101010101, -21846 = 0b1010101010101010
        torch.int32: [([0, -1], [0, 32]), ([1431655765, -1431655766], [16, 16])],  # 1431655765 = 0b010101... (32b), -1431655766 = 0b101010... (32b)
        torch.int64: [([0, -1], [0, 64]), ([6148914691236517205, -6148914691236517206], [32, 32])],  # 6148914691236517205 = 0b0101... (64b), -6148914691236517206 = 0b1010... (64b)
        torch.uint8: [([0, 255], [0, 8]), ([240, 15], [4, 4])],  # 240 = 0b11110000, 15 = 0b00001111
    }
    for dtype in dtypes:
        for idx, (vals, expected) in enumerate(test_cases[dtype]):
            t = torch.tensor(vals, dtype=dtype)
            result = bitwise_popcount(t)
            assert result.tolist() == expected, f"Failed for dtype {dtype}, test {idx}: got {result.tolist()}, expected {expected}"
            print(f"Passed dtype {dtype}, test {idx}: {vals} -> {result.tolist()}")
    print("All dtype tests passed.")
    print("-" * 30)

    # --- Efficiency test: large tensor ---
    print("Efficiency test on large tensor (CPU)...")
    large_tensor = torch.randint(0, 2**31 - 1, (10_000_000,), dtype=torch.int32)
    start = time.time()
    popcounts_large = bitwise_popcount(large_tensor)
    elapsed = time.time() - start
    print(f"CPU: Popcount on 10M int32 elements took {elapsed:.4f} seconds.")

    if torch.cuda.is_available():
        print("Efficiency test on large tensor (CUDA)...")
        large_tensor_cuda = large_tensor.to('cuda')
        torch.cuda.synchronize()
        start = time.time()
        popcounts_large_cuda = bitwise_popcount(large_tensor_cuda)
        torch.cuda.synchronize()
        elapsed_cuda = time.time() - start
        print(f"CUDA: Popcount on 10M int32 elements took {elapsed_cuda:.4f} seconds.")
        # Optional: check correctness
        assert torch.all(popcounts_large_cuda.cpu() == popcounts_large), "Mismatch between CPU and CUDA results!"
        print("CUDA and CPU results match.")
    else:
        print("CUDA not available, skipping CUDA efficiency test.")
    print("-" * 30)