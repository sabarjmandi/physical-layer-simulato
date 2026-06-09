import numpy as np
import matplotlib.pyplot as plt

# ============================================
# TEXT ↔ BITS
# ============================================

def text_to_bits(text):
    """Convert text to list of bits (ASCII 8-bit)"""
    return [int(b) for c in text for b in format(ord(c), '08b')]

def bits_to_text(bits):
    """Convert list of bits back to text"""
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

# ============================================
# LINE CODING
# ============================================

def nrz(bits):
    """NRZ (Non-Return to Zero) encoding: 1 → +1, 0 → -1"""
    return np.array([1 if b == 1 else -1 for b in bits])

def manchester(bits):
    """Manchester encoding: 1 → [+1, -1], 0 → [-1, +1]"""
    signal = []
    for b in bits:
        signal += [1, -1] if b == 1 else [-1, 1]
    return np.array(signal)

# ============================================
# BPSK MODULATION
# ============================================

def bpsk_mod(bits):
    """BPSK modulation: 1 → +1, 0 → -1"""
    return np.array([1 if b == 1 else -1 for b in bits])

def bpsk_demod(signal):
    """BPSK demodulation: positive → 1, negative → 0"""
    return [1 if s > 0 else 0 for s in signal]

# ============================================
# AWGN NOISE
# ============================================

def awgn(signal, snr_db):
    """Add AWGN (Additive White Gaussian Noise) to signal"""
    signal = np.array(signal)
    power = np.mean(signal**2)
    snr = 10**(snr_db/10)
    noise_power = power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise

# ============================================
# HAMMING CODE (7,4)
# ============================================

def hamming_encode(bits):
    """Hamming (7,4) encoding: 4 bits → 7 bits"""
    encoded = []
    for i in range(0, len(bits), 4):
        d = bits[i:i+4] + [0] * (4 - len(bits[i:i+4]))
        d1, d2, d3, d4 = d

        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4

        encoded += [p1, p2, d1, p3, d2, d3, d4]
    return encoded

def hamming_decode(bits):
    """Hamming (7,4) decoding with error correction"""
    decoded = []
    for i in range(0, len(bits), 7):
        b = bits[i:i+7]
        if len(b) < 7:
            continue

        p1, p2, d1, p3, d2, d3, d4 = b

        # Calculate syndrome
        s1 = p1 ^ d1 ^ d2 ^ d4
        s2 = p2 ^ d1 ^ d3 ^ d4
        s3 = p3 ^ d2 ^ d3 ^ d4

        error = s1 * 4 + s2 * 2 + s3

        # Correct error if found
        if error != 0:
            # Error positions in the 7-bit block
            pos = {1: 3, 2: 4, 3: 5, 4: 0, 5: 2, 6: 1, 7: 6}
            if error in pos:
                b[pos[error]] ^= 1

        decoded += [b[2], b[4], b[5], b[6]]
    return decoded

# ============================================
# BER CALCULATION
# ============================================

def ber(a, b):
    """Calculate Bit Error Rate between two bit sequences"""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    return sum(a[i] != b[i] for i in range(n)) / n

# ============================================
# PLOT SIGNALS
# ============================================

def plot_signals(original, noisy):
    """Plot original and noisy signals for comparison"""
    plt.figure(figsize=(12, 5))

    plt.subplot(2, 1, 1)
    plt.plot(original[:200], 'b-', linewidth=1.5)
    plt.title("Original Signal (Before Noise)", fontsize=12)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(noisy[:200], 'r-', linewidth=1.5)
    plt.title("Noisy Signal (After AWGN)", fontsize=12)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ============================================
# MAIN FUNCTION
# ============================================

def run():
    print("=" * 60)
    print("     PHYSICAL LAYER SIMULATOR")
    print("=" * 60)

    # Get input from user
    text = input("\n📝 Enter text: ")

    print("\n📡 Select Modulation/Encoding:")
    print("   1. NRZ (Non-Return to Zero)")
    print("   2. Manchester")
    print("   3. BPSK (Binary Phase Shift Keying)")

    mode = input("\n👉 Choose (1/2/3): ")

    snr = float(input("\n🔊 Enter SNR in dB (0-20, lower = more noise): "))

    # Step 1: Text to bits
    bits = text_to_bits(text)
    print(f"\n✅ Original bits: {''.join(map(str, bits))}")

    # Step 2: Hamming encoding
    bits = hamming_encode(bits)
    print(f"✅ After Hamming encoding: {len(bits)} bits")

    # Step 3: Modulation/Encoding
    if mode == "1":
        signal = nrz(bits)
        mod_name = "NRZ"
    elif mode == "2":
        signal = manchester(bits)
        mod_name = "Manchester"
    else:
        signal = bpsk_mod(bits)
        mod_name = "BPSK"

    print(f"✅ {mod_name} modulation applied")

    # Step 4: Add noise
    noisy_signal = awgn(signal, snr)
    print(f"✅ AWGN noise added (SNR = {snr} dB)")

    # Step 5: Demodulation
    received_bits = bpsk_demod(noisy_signal)
    print(f"✅ Demodulation complete")

    # Step 6: Hamming decoding
    received_bits = hamming_decode(received_bits)
    print(f"✅ Hamming decoding complete")

    # Step 7: Bits to text
    received_text = bits_to_text(received_bits)

    # Step 8: Calculate BER
    original_bits = text_to_bits(text)
    error_rate = ber(original_bits, received_bits) * 100

    # Results
    print("\n" + "=" * 60)
    print("                 RESULTS")
    print("=" * 60)
    print(f"📨 Original text: {text}")
    print(f"📬 Received text: {received_text}")
    print(f"📊 Bit Error Rate (BER): {error_rate:.2f}%")

    if received_text == text:
        print("\n✅ SUCCESS: Message received correctly!")
    elif error_rate < 20:
        print("\n⚠️ PARTIAL SUCCESS: Some errors but message is readable")
    else:
        print("\n❌ FAILED: Too much noise, message corrupted")

    print("=" * 60)

    # Plot signals
    plot_signals(signal, noisy_signal)


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    run()
