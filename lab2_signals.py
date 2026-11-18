import numpy as np
import matplotlib.pyplot as plt
import time
import math
import cmath
import fractions

operations = {'mul': 0, 'add': 0}

n = 8

def calc_Ak(f_vec, k, N):
    sum_val = 0
    global operations
    
    for m in range(N):
        angle = 2 * math.pi * k * m / N
        sum_val += f_vec[m] * math.cos(angle)
        
        operations['mul'] += 4
        operations['add'] += 1
        
    return sum_val

def calc_Bk(f_vec, k, N):
    sum_val = 0
    global operations
    
    for m in range(N):
        angle = 2 * math.pi * k * m / N
        sum_val += f_vec[m] * math.sin(angle)
        
        operations['mul'] += 4  
        operations['add'] += 1  
        
    return -sum_val

def calculate_Ck(f_vec, k, N):
    A_k = calc_Ak(f_vec, k, N)
    B_k = calc_Bk(f_vec, k, N)
    
    global operations
    operations['add'] += 1
    
    return complex(A_k, B_k)

def part2_data():
    N_2 = 96 + n
    bin_str = bin(N_2)[2:].zfill(7)
    bit_8 = '1' if n % 2 != 0 else '0'
    binary_8bit = bit_8 + bin_str[:7]

    s_nT = np.array([int(b) for b in binary_8bit])
    M = 8
    C_n = [calculate_Ck(s_nT, k, M) for k in range(M)]
    return M, s_nT, C_n, N_2, binary_8bit

def calc_s_t(t_array, Ck_scaled, M):
    s_val = np.zeros_like(t_array, dtype=float)

    s_val += Ck_scaled[0].real * np.ones_like(t_array)
    
    for i in range(1, M // 2):
        mag = abs(Ck_scaled[i])
        phi = cmath.phase(Ck_scaled[i])
        s_val += 2 * mag * np.cos((2 * np.pi * i * t_array) / M + phi)

    half = M // 2
    mag_half = abs(Ck_scaled[half])
    phi_half = cmath.phase(Ck_scaled[half])
    s_val += mag_half * np.cos((M * np.pi * t_array) / M + phi_half)
    
    return s_val


def part1():
    global operations
    operations = {'mul': 0, 'add': 0}

    N_1 = 32
    print(f"Кількість елементів N = {N_1}")

    np.random.seed(N_1)
    f_input = (np.random.random((1, N_1)) * 10)[0]
    print("Вхідний вектор сигналів:", np.round(f_input, 2))

    start_time = time.time()
    C_spectrum = [calculate_Ck(f_input, k, N_1) for k in range(N_1)]
    end_time = time.time()

    amplitudes = [abs(c) for c in C_spectrum]
    phases = [cmath.phase(c) for c in C_spectrum]

    print(f"Час обчислення ДПФ: {(end_time - start_time)*1000:.4f} мс")
    print(f"Кількість операцій: Множень = {operations['mul']}, Додавань = {operations['add']}")

    print("\nКомплексні коефіцієнти C_k:")
    for i, c in enumerate(C_spectrum):  
        print(f"C{i} = {c.real:.6f}{c.imag:+.6f}j; |C{i}| = {abs(c):.6f}; argC{i} = {cmath.phase(c)}*pi")

    # Графік спектру амплітуд
    plt.figure(figsize=(8, 5))
    plt.stem(range(N_1), amplitudes)
    plt.title("Спектр амплітуд")
    plt.xlabel("k")
    plt.ylabel("|C_k|")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    
    # Графік спектру фаз
    plt.figure(figsize=(8, 5))
    plt.stem(range(N_1), phases)
    plt.title("Спектр фаз")
    plt.xlabel("k")
    plt.ylabel("arg(C_k) [рад]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def part2():
    M, s_nT, C_n, N_2, binary_8bit = part2_data()

    print(f"Число N = {N_2}, Двійкове: {binary_8bit}")
    print("8 рівновіддалених відліків s(nT):", s_nT)

    print("\nКомплексні коефіцієнти C_n:")
    Ck_scaled = []
    for i, c in enumerate(C_n):
        c_scaled = c / M 
        Ck_scaled.append(c_scaled)
        
        modulus = abs(c_scaled)
        phase = cmath.phase(c_scaled)
        
        phase_pi = phase / math.pi
        if abs(phase_pi) < 1e-10:
            phase_str = "0"
        else:
            frac = fractions.Fraction(phase_pi).limit_denominator(100)
            if frac.denominator == 1:
                phase_str = str(frac.numerator)
            else:
                phase_str = f"{frac.numerator}/{frac.denominator}"

        print(f"C{i} = {c_scaled.real:.6f}{c_scaled.imag:+.6f}j; |C{i}| = {modulus:.6f}; argC{i} = {phase_str}*pi")

    t_analog = np.linspace(0, M, 1000)
    s_t = calc_s_t(t_analog, Ck_scaled, M)

    # Графік часової залежності відтвореного сигналу
    plt.figure(figsize=(8, 4))
    plt.plot(t_analog, s_t.real, label='Відновлений аналоговий сигнал s(t)', color='red')
    plt.stem(range(M), s_nT, basefmt=" ", label='Початкові відліки s(nT)')
    plt.title("Відтворення аналогового сигналу")
    plt.xlabel("Час t")
    plt.ylabel("Амплітуда")
    plt.legend()
    plt.grid(True)
    plt.show()


def part3():
    M, s_nT, C_n, _, _ = part2_data()
    s_reconstructed = []
    for m in range(M):
        s_m = complex(0, 0)
        for k in range(M):
            angle = 2 * math.pi * k * m / M
            s_m += (C_n[k] / M) * complex(math.cos(angle), math.sin(angle))
        s_reconstructed.append(s_m.real)

    print("\nПочаткові відліки: ", s_nT)
    print("Відновлені значення відліків ОДПФ:")
    print(np.round(s_reconstructed, 2))
    
    print("\nЗначення відліків s(nTδ) n=(0, N-1):")
    for i, val in enumerate(s_reconstructed):
        print(f"s({i}Tδ) = {val:.6f}")

if __name__ == "__main__":
    part1()
    # part2()
    # part3()