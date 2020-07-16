package com.s0lver;

public class Layer {
    private final Neuron[] neurons;

    /**
     * Constructor used for the hidden and output layers.
     *
     * @param inputWeightsPerNeuron The number of input weights per neuron.
     * @param numberNeurons         The number of neurons to create in the layer.
     */
    public Layer(int inputWeightsPerNeuron, int numberNeurons) {
        this.neurons = new Neuron[numberNeurons];
        for (int i = 0; i < numberNeurons; i++) {
            double[] weights = new double[inputWeightsPerNeuron];
            for (int j = 0; j < inputWeightsPerNeuron; j++) {
                weights[j] = Utils.generateRandomDouble(Neuron.minWeightValue, Neuron.maxWeightValue);
            }
            neurons[i] = new Neuron(weights, Utils.generateRandomDouble(0, 1));
        }
    }

    /**
     * Constructor used for the input layer
     *
     * @param input The input data for the neurons in input layer.
     *              The input length defines the number of neurons in the input layer.
     */
    public Layer(double[] input) {
        this.neurons = new Neuron[input.length];
        for (int i = 0; i < input.length; i++) {
            this.neurons[i] = new Neuron(input[i]);
        }
    }

    public Neuron[] getNeurons() {
        return neurons;
    }

    public Neuron getNeuron(int index) {
        return neurons[index];
    }

    public int getNumOfNeurons() {
        return neurons.length;
    }
}
