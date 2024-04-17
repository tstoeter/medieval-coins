import os

base_path, _ = os.path.split(os.path.dirname(__file__))
base_path    = base_path + "/"
data_path    = base_path + "data/"
outp_path    = base_path + "output/"
rslt_path    = base_path + "results/"

coin_templates = ["9903_12_203_2022-01-12_14-27-46_",   # Bahrfeldt 19
                  "8818_04_169e_2021-03-12_15-37-03_",  # Mehl 499
                  "9653_10_133_2021-07-30_14-38-17_"]   # Mehl 595

quantile = 0.76
accum_idx = 4

if __name__ == "__main__":
    print(base_path, data_path, outp_path, rslt_path)
    print(coin_templates)
