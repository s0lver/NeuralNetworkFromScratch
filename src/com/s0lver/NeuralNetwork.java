package com.s0lver;

import java.util.Arrays;
import java.util.List;

public class NeuralNetwork {
    private final Layer[] layers;

    private final Layer outputLayer;

    static {
        Neuron.setRangeWeight(0.0, 1.0);
    }

    public NeuralNetwork(List<int[]> layerDimensions) {
        this.layers = new Layer[layerDimensions.size()];
        for (int i = 0; i < layerDimensions.size(); i++) {
            var dim = layerDimensions.get(i);
            layers[i] = new Layer(dim[0], dim[1]);
        }

        this.outputLayer = layers[layers.length - 1];
    }

    /**
     * Performs the forward (or predict for that matter) for the network using the input data.
     *
     * @param input An input (n-dimensional vector) for the network.
     */
    public double[] forward(double[] input) {
        // First, input to input layer
        layers[0] = new Layer(input);

        // Then forward for the rest of layers
        for (int i = 1; i < layers.length; i++) {
            for (int j = 0; j < layers[i].getNumOfNeurons(); j++) {
                double sum = 0.0;
                for (int k = 0; k < layers[i - 1].getNumOfNeurons(); k++) {
                    sum += layers[i - 1].getNeuron(k).getValue() * layers[i].getNeuron(j).getWeight(k);
                }
                layers[i].getNeuron(j).setValue(Utils.Sigmoid(sum));
            }
        }

        return Arrays.stream(outputLayer.getNeurons()).mapToDouble(Neuron::getValue).toArray();
    }

    /**
     * Performs the training of the network
     *
     * @param trainingDataset    The training dataset to use
     * @param trainingIterations The number of iterations to perform.
     * @param learningRate       The learning rate for updating weights.
     */
    public void train(TrainingData[] trainingDataset, int trainingIterations, double learningRate) {
        for (int i = 0; i < trainingIterations; i++) {
            for (TrainingData trainingData : trainingDataset) {
                forward(trainingData.getData());
                backward(learningRate, trainingData);
            }
        }
    }

    private void backward(double learningRate, TrainingData trainingData) {
        int numOfLayers = layers.length;
        int outIndex = numOfLayers - 1;

        // Updating the output layers, for each output neuron
        for (int i = 0; i < layers[outIndex].getNumOfNeurons(); i++) {
            double output = layers[outIndex].getNeuron(i).getValue();
            double target = trainingData.getExpectedOutput()[i];
            double derivative = output - target;
            double delta = derivative * (output * (1 - output));
            layers[outIndex].getNeuron(i).setGradient(delta);

            // for each weight
            for (int j = 0; j < layers[outIndex].getNeuron(i).getNumOfWeights(); j++) {
                double previousOutput = layers[outIndex - 1].getNeuron(j).getValue();
                double error = delta * previousOutput;
                layers[outIndex].getNeuron(i).setCacheWeight(j, layers[outIndex].getNeuron(i).getWeight(j) - learningRate * error);
            }
        }

        // Update subsequent layers
        for (int i = outIndex - 1; i > 0; i--) { // We don't process layer 0 as it is the input layer
            // For all neurons in layer
            for (int j = 0; j < layers[i].getNumOfNeurons(); j++) {
                double output = layers[i].getNeuron(j).getValue();
                double gradientSum = sumGradient(j, i + 1);
                double delta = (gradientSum) * (output * (1 - output));
                layers[i].getNeuron(j).setGradient(delta);

                // For all its weights
                for (int k = 0; k < layers[i].getNeuron(j).getNumOfWeights(); k++) {
                    double previousOutput = layers[i - 1].getNeuron(k).getValue();
                    double error = delta * previousOutput;
                    layers[i].getNeuron(j).setCacheWeight(k, layers[i].getNeuron(j).getWeight(k) - learningRate * error);
                }
            }
        }
        // Finally update the weights
        for (int i = 0; i < layers.length; i++) {
            for (int j = 0; j < layers[i].getNumOfNeurons(); j++) {
                layers[i].getNeuron(j).updateWeight();
            }
        }
    }

    public double sumGradient(int nIndex, int lIndex) {
        double gradientSum = 0;
        Layer currentLayer = layers[lIndex];
        for (int i = 0; i < currentLayer.getNumOfNeurons(); i++) {
            Neuron currentNeuron = currentLayer.getNeuron(i);
            gradientSum += currentNeuron.getWeight(nIndex) * currentNeuron.getGradient();
        }
        return gradientSum;
    }
}
