//
//  Layer.cpp
//  test1203
//
//  Created by 许清嘉 on 12/5/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#include "Layer.hpp"
#include "assert.h"

Layer::Layer(Layer* prev) : prev(prev){
    
}

Layer::Layer(const int nodeNum, Layer* prev) : nodeNum(nodeNum), prev(prev){
    assert(nodeNum > 0);
    layerType = layerSet::INPUT;
}
