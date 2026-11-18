import time
import cmath
import numpy as np
import lab2_signals

class FastFourierTransform:
    def __init__(self):
        self.fft_add = 0
        self.fft_mul = 0

    def fft(self, x):
        N = len(x)
        if N <= 1:
            return x

        even = self.fft(x[0::2])
        odd = self.fft(x[1::2])

        T = []
        for k in range(N // 2):
            factor = cmath.exp(-2j * cmath.pi * k / N) * odd[k]
            T.append(factor)
            self.fft_mul += 1

        self.fft_add += N

        return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

def main():
    N = 32
    
    np.random.seed(N)
    f_input = (np.random.random((1, N)) * 10)[0]
    print("Вхідний вектор сигналів:", np.round(f_input, 2))
    print(f"\nКількість елементів N = {N}")

    transform = FastFourierTransform()

    start_time = time.perf_counter()
    fft_result = transform.fft(f_input)
    end_time = time.perf_counter()

    print("\n--- Результати ШПФ ---")
    print(f"\nЧас обчислення ШПФ: {(end_time - start_time) * 1000:.4f} мс")
    print(f"Кількість операцій ШПФ: Множень = {transform.fft_mul}, Додавань = {transform.fft_add}")
    
    print("\nКомплексні коефіцієнти C_k (ШПФ) (перші 5):")
    for i in range(5):
        c = fft_result[i]
        phase_pi = cmath.phase(c) / cmath.pi
        print(f"C{i} = {c.real:.6f}{c.imag:+.6f}j; |C{i}| = {abs(c):.6f}; argC{i} = {phase_pi:.6f}*pi")


    lab2_signals.operations = {'mul': 0, 'add': 0}
    
    start_time_dft = time.perf_counter()
    dft_result = [lab2_signals.calculate_Ck(f_input, k, N) for k in range(N)]
    end_time_dft = time.perf_counter()
    time_dft_ms = (end_time_dft - start_time_dft) * 1000
    
    print("\n--- Результати ДПФ ---")
    print(f"\nЧас обчислення ДПФ: {time_dft_ms:.4f} мс")
    print(f"Кількість операцій ДПФ: Множень = {lab2_signals.operations['mul']}, Додавань = {lab2_signals.operations['add']}")
    
    print("\nКомплексні коефіцієнти C_k (перші 5):")
    for i in range(5):
        c = dft_result[i]
        phase_pi = cmath.phase(c) / cmath.pi
        print(f"C{i} = {c.real:.6f}{c.imag:+.6f}j; |C{i}| = {abs(c):.6f}; argC{i} = {phase_pi:.6f}*pi")

    print("\n" + "="*50)

if __name__ == "__main__":
    main()