An implementation of support vector machines (SVM) using C++ was developed from scratch. 
Additionally, a Python interface was created to mimic the interface offered by scikit-learn for SVM. 
The algorithm from the paper "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines" was employed for solving the dual problem.

# Build: <br />
$ git clone https://github.com/mpountoui/Support-Vector-Machine.git <br />
$ cd Support-Vector-Machine <br />
$ mkdir build <br />
$ cd build <br />
$ cmake .. -D PYTHON_LIBRARY_DIR=".../Support-Vector-Machine/venv/lib/python3.9/site-packages" <br />
$ make <br />
$ make install <br />

# Examples: <br /> <br />
| Kernel  | C  |
| ------- | -- |
| linear  | 10 |
![linear_c_10](https://github.com/mpountoui/Support-Vector-Machine/assets/119242445/57f4a5cd-4718-4b71-89e9-083e534cbf93)
<br />
<br />

Kernel : rbf <br />
C : 10 <br />
gamma : 1
![rbf_c_10_g_1](https://github.com/mpountoui/Support-Vector-Machine/assets/119242445/f84e4331-69a4-478f-9400-f7d38d9a1984)

Kernel : poly <br />
C : 10 <br />
gamma  : 1
degree : 2
coef0  : 1.0
![poly_c_10_g_1_d_2_cf_1](https://github.com/mpountoui/Support-Vector-Machine/assets/119242445/ba3c6633-8a45-46ad-899c-ce5c87518690)
