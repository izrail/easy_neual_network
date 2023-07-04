#include <iostream>
#include "conv2d/conv2d.hpp"
using namespace std;

int main() {
    Conv2d conv1(1, 10, 10, 3, 1, 3, 3);
    conv1.init();
    Op *op = &conv1;
    op->run();
    conv1.result();
    return 0;

}
