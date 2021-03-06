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
        this.neurons[0] = new Neuron();

        for (int i = 1; i < numberNeurons; i++) {
            double[] weights = new double[inputWeightsPerNeuron];
            for (int j = 0; j < inputWeightsPerNeuron; j++) {
                weights[j] = Utils.generateRandomDouble(Neuron.minWeightValue, Neuron.maxWeightValue);
            }
            neurons[i] = new Neuron(weights);
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
