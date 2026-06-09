import numpy as np
import matplotlib.pyplot as plt

# TEXT ↔ BITS
def text_to_bits(text):
    return [int(b) for c in text for b in format(ord(c), '08b')]

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

# LINE CODING
def nrz(bits):
    return np.array([1 if b == 1 else -1 for b in bits])

def manchester(bits):
    signal = []
    for b in bits:
        signal += [1, -1] if b == 1 else [-1, 1]
    return np.array(signal)

# BPSK
def bpsk_mod(bits):
    return np.array([1 if b == 1 else -1 for b in bits])

def bpsk_demod(signal):
    return [1 if s > 0 else 0 for s in signal]

# AWGN
def awgn(signal, snr_db):
    signal = np.array(signal)
    power = np.mean(signal**2)
    snr = 10**(snr_db/10)
    noise_power = power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise

# HAMMING (7,4)
def hamming_encode(bits):
    encoded = []
    for i in range(0, len(bits), 4):
        d = bits[i:i+4] + [0]*(4 - len(bits[i:i+4]))
        d1,d2,d3,d4 = d

        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4

        encoded += [p1, p2, d1, p3, d2, d3, d4]
    return encoded

def hamming_decode(bits):
    decoded = []
    for i in range(0, len(bits), 7):
        b = bits[i:i+7]
        if len(b) < 7:
            continue

        p1,p2,d1,p3,d2,d3,d4 = b

        s1 = p1 ^ d1 ^ d2 ^ d4
        s2 = p2 ^ d1 ^ d3 ^ d4
        s3 = p3 ^ d2 ^ d3 ^ d4

        error = s1*4 + s2*2 + s3

        if error != 0:
            pos = {1:3,2:4,3:5,4:0,5:2,6:1,7:3}
            if error in pos:
                b[pos[error]] ^= 1

        decoded += [b[2], b[4], b[5], b[6]]
    return decoded

# BER
def ber(a, b):
    n = min(len(a), len(b))
    return sum(a[i] != b[i] for i in range(n)) / n

# PLOT
def plot(sig, noisy):
    plt.figure(figsize=(10,4))

    plt.subplot(2,1,1)
    plt.plot(sig[:200])
    plt.title("Original Signal")

    plt.subplot(2,1,2)
    plt.plot(noisy[:200])
    plt.title("Noisy Signal")

    plt.tight_layout()
    plt.show()

# RUN
def run():
    text = input("Enter text: ")

    print("1 NRZ")
    print("2 Manchester")
    print("3 BPSK")
    mode = input("Choose: ")

    snr = float(input("SNR: "))

    bits = text_to_bits(text)
    bits = hamming_encode(bits)

    if mode == "1":
        signal = nrz(bits)
    elif mode == "2":
        signal = manchester(bits)
    else:
        signal = bpsk_mod(bits)

    noisy = awgn(signal, snr)

    rec_bits = bpsk_demod(noisy)
    rec_bits = hamming_decode(rec_bits)

    rec_text = bits_to_text(rec_bits)

    error = ber(bits, rec_bits)

    print("Original:", text)
    print("Received:", rec_text)
    print("BER:", round(error*100, 2), "%")

    plot(signal, noisy)

if __name__ == "__main__":
    run()
