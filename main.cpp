//
//  main.cpp
//  test1203
//
//  Created by 许清嘉 on 12/3/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#include "Model.hpp"
#include <ctime>

void test1(){
    cout << "========================\n\t\tTEST 1\n========================" << endl;
    clock_t start_train1, end_train1;
    Model m1;
    m1.addInputLayer(10);
    m1.addConvLayer(2, 3, 1, ConvLayer::valid, ConvLayer::sigmoid);
    m1.addPoolLayer(2, PoolLayer::general, PoolLayer::max);
    m1.addFcLayer(2, FcLayer::softmax);
    m1.testInitialize1();
    start_train1 = clock();
    m1.train(vector<float>({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}), vector<float>({0, 1}), 0.5, 2, FcLayer::cross_entropy);
    end_train1 = clock();
    cout << "···Training consumes " << (double)(end_train1 - start_train1) / CLOCKS_PER_SEC << "s.···" << endl << endl;
}

void test2(){
    cout << "========================\n\t\tTEST 2\n========================" << endl;
    clock_t start_train2, end_train2;
    Model m2;
    m2.addInputLayer(6);
    m2.addConvLayer(2, 3, 1, ConvLayer::valid, ConvLayer::sigmoid);
    m2.addPoolLayer(2, PoolLayer::general, PoolLayer::average);
    m2.addConvLayer(3, 3, 1, ConvLayer::valid, ConvLayer::sigmoid);
    m2.addPoolLayer(2, PoolLayer::general, PoolLayer::max);
    m2.addFcLayer(2, FcLayer::softmax);
    m2.testInitialize2();
    start_train2 = clock();
    m2.train(vector<float>({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}), vector<float>({1, 0}), 0.7, 2, FcLayer::cross_entropy);
    end_train2 = clock();
    cout << "···Training consumes " << (double)(end_train2 - start_train2) / CLOCKS_PER_SEC << "s.···" << endl << endl;
}

void test3(){
    cout << "========================\n\t\tTEST 3\n========================" << endl;
    clock_t start_train3, end_train3;
    Model m3;
    m3.addInputLayer(1);
    m3.addConvLayer(2, 3, 2, ConvLayer::full, ConvLayer::sigmoid);
    m3.addPoolLayer(2, PoolLayer::general, PoolLayer::max);
    m3.addFcLayer(2, FcLayer::sigmoid);
    m3.addFcLayer(2, FcLayer::softmax);
    m3.testInitialize3();
    start_train3 = clock();
    m3.train(vector<float>({0.1}), vector<float>({1, 0}), 0.5, 2, FcLayer::cross_entropy);
    end_train3 = clock();
    cout << "···Training consumes " << (double)(end_train3 - start_train3) / CLOCKS_PER_SEC << "s.···" << endl << endl;
}

void test4(){
    cout << "========================\n\t\tTEST 4\n========================" << endl;
    clock_t start_train4, end_train4;
    Model m4;
    m4.addInputLayer(3);
    m4.addConvLayer(2, 3, 1, ConvLayer::same, ConvLayer::ReLU);
    m4.addPoolLayer(3, PoolLayer::general, PoolLayer::average);
    m4.addFcLayer(2, FcLayer::softmax);//sigmoid
    m4.testInitialize4();
    start_train4 = clock();
    m4.train(vector<float>({0.1, 0.2, 0.3}), vector<float>({1, 0}), 0.5, 2, FcLayer::cross_entropy);//squared
    end_train4 = clock();
    cout << "···Training consumes " << (double)(end_train4 - start_train4) / CLOCKS_PER_SEC << "s.···" << endl << endl;
}

void test5(){
    cout << "========================\n\t\tTEST 5\n========================" << endl;
    clock_t start_train5, end_train5;
    Model m5;
    m5.addInputLayer(3);
    m5.addConvLayer(2, 3, 2, ConvLayer::same, ConvLayer::ReLU);
    m5.addPoolLayer(3, PoolLayer::overlap, PoolLayer::average, 1);
    m5.addFcLayer(2, FcLayer::softmax);
    m5.testInitialize4();
    start_train5 = clock();
    m5.train(vector<float>({0.1, 0.2, 0.3}), vector<float>({1, 0}), 0.5, 2, FcLayer::cross_entropy);
    end_train5 = clock();
    cout << "···Training consumes " << (double)(end_train5 - start_train5) / CLOCKS_PER_SEC << "s.···" << endl << endl;
}

void test6(){
    cout << "========================\n\t\tTEST 6\n========================" << endl;
    clock_t start_train6, end_train6;
    Model m6;
    m6.addInputLayer(3);
    m6.addConvLayer(2, 3, 2, ConvLayer::same, ConvLayer::ReLU);
    m6.addPoolLayer(3, PoolLayer::overlap, PoolLayer::max, 2);
    m6.addFcLayer(2, FcLayer::softmax);
    m6.testInitialize4();
    start_train6 = clock();
    m6.train(vector<float>({0.1, 0.2, 0.3}), vector<float>({1, 0}), 0.5, 2, FcLayer::cross_entropy);
    end_train6 = clock();
    cout << "···Training consumes " << (double)(end_train6 - start_train6) / CLOCKS_PER_SEC << "s.···" << endl << endl;
}

void test7(){
    cout << "========================\n\t\tTEST 7\n========================" << endl;
    clock_t start_train7, end_train7;
    Model m7;
    m7.addInputLayer(3);
    m7.addConvLayer(2, 3, 1, ConvLayer::full, ConvLayer::ReLU);
    m7.addPoolLayer(2, PoolLayer::general, PoolLayer::max);
    m7.addConvLayer(2, 3, 2, ConvLayer::valid, ConvLayer::ReLU);
    m7.addPoolLayer(2, PoolLayer::general, PoolLayer::average);
    m7.addFcLayer(2, FcLayer::softmax);
    m7.testInitialize7();
    start_train7 = clock();
    m7.train(vector<float>({0.1, 0.2, 0.3}), vector<float>({1, 0}), 0.5, 2, FcLayer::cross_entropy);
    end_train7 = clock();
    cout << "···Training consumes " << (double)(end_train7 - start_train7) / CLOCKS_PER_SEC << "s.···" << endl << endl;
}

void test8(){
    cout << "========================\n\t\tTEST 8\n========================" << endl;
    clock_t start_train8, end_train8;
    Model m8;
    m8.addInputLayer(4);
    m8.addConvLayer(1, 3, 1, ConvLayer::same, ConvLayer::ReLU);
    m8.addPoolLayer(2, PoolLayer::overlap, PoolLayer::max, 1);
    m8.addPoolLayer(2, PoolLayer::overlap, PoolLayer::max, 1);
    m8.addFcLayer(2, FcLayer::softmax);
    m8.testInitialize8();
    start_train8 = clock();
    m8.train(vector<float>({0.1, 0.2, 0.3, 0.4}), vector<float>({1, 0}), 0.5, 2, FcLayer::cross_entropy);
    end_train8 = clock();
    cout << "···Training consumes " << (double)(end_train8 - start_train8) / CLOCKS_PER_SEC << "s.···" << endl << endl;
}

void test9(){
    cout << "========================\n\t\tTEST 9\n========================" << endl;
    clock_t start_train9, end_train9;
    Model m9;
    m9.addInputLayer(1);
    m9.addConvLayer(2, 3, 2, ConvLayer::full, ConvLayer::sigmoid);
    m9.addPoolLayer(2, PoolLayer::general, PoolLayer::max);
    m9.addConvLayer(2, 3, 1, ConvLayer::same, ConvLayer::sigmoid);
    m9.addPoolLayer(3, PoolLayer::overlap, PoolLayer::average, 1);
    m9.addFcLayer(2, FcLayer::softmax);
    m9.testInitialize9();
    start_train9 = clock();
    m9.train(vector<float>({0.1}), vector<float>({1, 0}), 0.5, 2, FcLayer::cross_entropy);
    end_train9 = clock();
    cout << "···Training consumes " << (double)(end_train9 - start_train9) / CLOCKS_PER_SEC << "s.···" << endl << endl;
}

void test10(){
    cout << "========================\n\t\tTEST 10\n========================" << endl;
    clock_t start_train10, end_train10;
    Model m10;
    m10.addInputLayer(4);
    m10.addPoolLayer(2, PoolLayer::overlap, PoolLayer::max, 1);
    m10.addPoolLayer(2, PoolLayer::overlap, PoolLayer::max, 1);
    m10.addFcLayer(2, FcLayer::softmax);
    m10.testInitialize10();
    start_train10 = clock();
    m10.train(vector<float>({0.1, 0.2, 0.3, 0.4}), vector<float>({1, 0}), 0.5, 2, FcLayer::cross_entropy);
    end_train10 = clock();
    cout << "···Training consumes " << (double)(end_train10 - start_train10) / CLOCKS_PER_SEC << "s.···" << endl << endl;
}

void test11(){
    cout << "========================\n\t\tTEST 11\n========================" << endl;
    clock_t start_train11, end_train11;
    Model m11;
    m11.addInputLayer(2);
    m11.addFcLayer(2, FcLayer::softmax);
    m11.testInitialize11();
    start_train11 = clock();
    m11.train(vector<float>({0.1, 0.2}), vector<float>({1, 0}), 0.5, 2, FcLayer::cross_entropy);
    end_train11 = clock();
    cout << "···Training consumes " << (double)(end_train11 - start_train11) / CLOCKS_PER_SEC << "s.···" << endl << endl;
}

void test12(){
    cout << "========================\n\t\tTEST 12\n========================" << endl;
    clock_t start_train12, end_train12;
    Model m12;
    m12.addInputLayer(2);
    m12.addConvLayer(2, 3, 1, ConvLayer::same, ConvLayer::ReLU);
    m12.addConvLayer(1, 3, 1, ConvLayer::valid, ConvLayer::ReLU);
    m12.addFcLayer(2, FcLayer::softmax);
    m12.testInitialize12();
    start_train12 = clock();
    m12.train(vector<float>({0.1, 0.2}), vector<float>({1, 0}), 0.5, 2, FcLayer::cross_entropy);
    end_train12 = clock();
    cout << "···Training consumes " << (double)(end_train12 - start_train12) / CLOCKS_PER_SEC << "s.···" << endl << endl;
}

void test13(){
    cout << "========================\n\t\tTEST 13\n========================" << endl;
    clock_t start_train13, end_train13;
    Model m13;
    m13.addInputLayer(2);
    m13.addFcLayer(4, FcLayer::sigmoid);
    m13.addPoolLayer(3, PoolLayer::overlap, PoolLayer::max, 1);
    m13.addFcLayer(2, FcLayer::softmax);
    m13.testInitialize13();
    start_train13 = clock();
    m13.train(vector<float>({0.1, 0.2}), vector<float>({1, 0}), 0.7, 2, FcLayer::cross_entropy);
    end_train13 = clock();
    cout << "···Training consumes " << (double)(end_train13 - start_train13) / CLOCKS_PER_SEC << "s.···" << endl << endl;
}

void test14(){
    cout << "========================\n\t\tTEST 14\n========================" << endl;
    clock_t start_train14, end_train14;
    Model m14;
    m14.addInputLayer(2);
    m14.addFcLayer(3, FcLayer::ReLU);
    m14.addConvLayer(2, 3, 1, ConvLayer::valid, ConvLayer::ReLU);
    m14.addFcLayer(2, FcLayer::softmax);
    m14.testInitialize14();
    start_train14 = clock();
    m14.train(vector<float>({0.1, 0.2}), vector<float>({1, 0}), 0.3, 2, FcLayer::cross_entropy);
    end_train14 = clock();
    cout << "···Training consumes " << (double)(end_train14 - start_train14) / CLOCKS_PER_SEC << "s.···" << endl << endl;
}

void test15(){
    cout << "========================\n\t\tTEST 15\n========================" << endl;
    clock_t start_train15, end_train15;
    Model m15;
    m15.addInputLayer(2);
    m15.addFcLayer(3, FcLayer::ReLU);
    m15.addConvLayer(2, 3, 1, ConvLayer::valid, ConvLayer::ReLU);
    m15.addFcLayer(2, FcLayer::softmax);
    start_train15 = clock();
    m15.train(vector<float>({0.1, 0.2}), vector<float>({1, 0}), 0.3, 2, FcLayer::cross_entropy);
    end_train15 = clock();
    cout << "···Training consumes " << (double)(end_train15 - start_train15) / CLOCKS_PER_SEC << "s.···" << endl << endl;
}

int main(int argc, const char * argv[]) {
    // insert code here...

    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
    test7();
    test8();
    test9();
    test10();
    test11();
    test12();
    test13();
    test14();
    test15();
    
    return 0;
}
