def binary_to_real(binary):
    v = 0
    for i in range(len(binary)):
        v += binary[i] * (2 ** (- i - 1))
    if v > 0:
        return v
    else:
        return v + 0.0000000001


def binary_to_integer(binary):
    v = 0
    for i in range(len(binary)):
        v += binary[len(binary) - i - 1] * 2**(i)
    if v > 0:
        return int(v)
    else:
        return int(1)