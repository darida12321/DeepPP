<div align="center">
    <img src="doc/logo.png">
    <h1> Deeppp </h1>
    <h3>deep neural network with c++ metaprogramming</h3>
    </br>
</div>

## About the Project 
---
Deeppp is a simple library to generate your own deep neural networks. We aim to utilise template metaprogramming to increase performance as compared to standard neural networks implemented without templates. 

## Getting Started 
---

### Prerequisites

To use Deeppp, we need to use the eigen library. For the testing and benchmarking, we use the libraries google test and google benchmark. You can install these by running the ```configure.sh``` script as below. 
```bash
./scripts/configure.sh 
```

### Build
You can build the neural network library the ```build.sh``` script by running the script as below. 
```bash
./scripts/build.sh
```

### Test
We have some pre-existing tests to check if everything is working as expected. You can manually run these by running the `test.sh` script. 
```bash
./scripts/test.sh
```

## Usage
---
To build a neural network in your own project, you need to first define the structure of your network. We have two ways you can do this, 
1. Using a vector of weights and biases
```cpp
Network(std::vector<MatrixXd> weights, std::vector<VectorXd> biases,
        std::vector<ActivationFunction*> act_func);
```
2. Using a vector to define the size of each layer
```cpp
Network(std::vector<int> layer_sizes,
        std::vector<ActivationFunction*> act_func);
```

We have some pre-existing `ActivationFunction` objects you may want to use, but feel free to make your own by inheriting the `ActivationFunction` class and defining your own function and derivative. Our pre-existing functions are 
```cpp
Sigmoid sigmoid;
Softmax softmax;
Relu relu;
Linear linear;
```
Once you have setup your Neural Network, you can train the network by calling `train` on your input and output vectors. You will also need to specifiy an amount by which you want the weights and biases to be updated while training. 
```cpp
void train(std::vector<VectorXd>, std::vector<VectorXd>, double);
```
Finally, you may get predictions by `forwardProp` which propogates your input vector through the neural network. 
```cpp
VectorXd forwardProp(VectorXd in);
```

## Example 
---
In this example, we create a simple 1-1 neural network and train it on simple data. 

```cpp
// Create 1-1 neural network
MatrixXd w(2, 2);
w << 1, 1, 1, 1;
VectorXd b(2);
b << 1, 1;
Network network(std::vector<MatrixXd>{w, w}, std::vector<VectorXd>{b, b},
                std::vector<ActivationFunction*>{&relu, &relu}

// Create example data point
VectorXd in1(2);
in1 << 1.0, 1.0;
VectorXd out1(2);
out1 << 3.0, 3.0;
VectorXd in2(2);
in2 << 3.0, 5.0;
VectorXd out2(2);
out2 << 6.0, 8.0;
std::vector<VectorXd> input{in1, in2};
std::vector<VectorXd> output{out1, out2};

// Train the network
network.train(input, output, 1);
```
