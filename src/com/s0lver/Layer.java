package com.s0lver;

public class Layer {
    private final Neuron[] neurons;
    private Neuron biasNeuron;

    /**
     * Constructor
     *
     * @param inputWeightsPerNeuron The number of input weights per neuron.
     * @param numberNeurons         The number of neurons to create in the layer.
     */
    public Layer(int inputWeightsPerNeuron, int numberNeurons, boolean createBiasNeuron) {
        if (createBiasNeuron) {
            this.biasNeuron = new Neuron();
        }
        this.neurons = new Neuron[numberNeurons];
        for (int i = 0; i < numberNeurons; i++) {
            double[] weights;
            final double biasWeight = Utils.generateRandomDouble(Neuron.minWeightValue, Neuron.maxWeightValue);
            if (inputWeightsPerNeuron > 0) {
                weights = generateRandomWeights(inputWeightsPerNeuron);
                neurons[i] = new Neuron(biasWeight, weights);
            } else {
                neurons[i] = new Neuron(biasWeight);
            }
        }
    }

    private double[] generateRandomWeights(int numOfWeights) {
        double[] weights = new double[numOfWeights];
        for (int j = 0; j < numOfWeights; j++) {
            weights[j] = Utils.generateRandomDouble(Neuron.minWeightValue, Neuron.maxWeightValue);
        }
        return weights;
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

    public Neuron getBiasNeuron() {
        return biasNeuron;
    }
}
