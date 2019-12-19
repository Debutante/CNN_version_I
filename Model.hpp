//
//  Model.hpp
//  test1203
//
//  Created by 许清嘉 on 12/8/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#ifndef Model_hpp
#define Model_hpp

#define TEST 0
#define NORMAL 1

#include <iostream>
#include <vector>

#include "ConvLayer.hpp"
#include "PoolLayer.hpp"
#include "FcLayer.hpp"

class Model{
public:
    vector<Layer*> net;
    FcLayer::lossSet lossFunction;
    vector<float> output;
    int currentIteration;
    int maxIteration;
    float learningRate;
    int startLayerNo;
    int mode;
    
    Model();
    Model(const vector<Layer*>& net);
    void addInputLayer(const int);
    void addConvLayer(const int, const int, const int, const ConvLayer::convSet&, const ConvLayer::activationSet&);
    void addPoolLayer(const int, const PoolLayer::poolSet&, const PoolLayer::sampleSet&);
    void addPoolLayer(const int, const PoolLayer::poolSet&, const PoolLayer::sampleSet&, const int);
    void addFcLayer(const int, const FcLayer::activationSet&);
    void testInitialize1();
    void testInitialize2();
    void testInitialize3();
    void testInitialize4();
    void testInitialize7();
    void testInitialize8();
    void testInitialize9();
    void testInitialize10();
    void testInitialize11();
    void testInitialize12();
    void testInitialize13();
    void testInitialize14();

    
    void train(const vector<float>&, const vector<float>&, const float, const int, const FcLayer::lossSet&);
    ~Model();
    
private:
    void paramCheck();
    void initialize();
    void forwardPropagation();
    void crossEntropyLoss();
    void squaredLoss();
    void logCoshLoss();
    void lossComputation();
    void coefComputation();
    void backwardPropagation();
    
};

#endif /* Model_hpp */
