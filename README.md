An implementation of support vector machines (SVM) using C++ was developed from scratch. 
Additionally, a Python interface was created to mimic the interface offered by scikit-learn for SVM. 
The algorithm from the paper "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines" was employed for solving the dual problem.

Build: <br />
$ git clone https://github.com/mpountoui/Support-Vector-Machine.git <br />
$ cd Support-Vector-Machine <br />
$ mkdir build <br />
$ cd build <br />
$ cmake .. -D PYTHON_LIBRARY_DIR=".../Support-Vector-Machine/venv/lib/python3.9/site-packages" <br />
$ make <br />
$ make install <br />
