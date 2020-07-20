package com.s0lver.neuralnetwork;

/**
 * This class models a neural network's layer, composed by neurons and optionally with a bias neuron.
 */
public class Layer {
    /**
     * The list of neurons (excepting the bias neuron) for the layer
     */
    private final Neuron[] neurons;

    /**
     * The bias neuron for the layer
     */
    private Neuron biasNeuron;

    /**
     * Constructor
     *
     * @param inputWeightsPerNeuron The number of input weights per neuron. It should be specified as 0 in case the created layer is the input layer.
     * @param numberNeurons         The number of neurons to create in the layer.
     * @param createBiasNeuron      Whether the bias neuron for this layer should be created
     */
    public Layer(int inputWeightsPerNeuron, int numberNeurons, boolean createBiasNeuron) {
        if (createBiasNeuron) {
            this.biasNeuron = new Neuron();
        }

        this.neurons = new Neuron[numberNeurons];
        for (int i = 0; i < numberNeurons; i++) {
            double[] neuronWeights;
            final double biasWeight = Utils.generateRandomDouble(Neuron.minWeightValue, Neuron.maxWeightValue);
            if (inputWeightsPerNeuron > 0) {
                neuronWeights = Utils.generateRandomWeights(inputWeightsPerNeuron);
                neurons[i] = new Neuron(biasWeight, neuronWeights);
            } else {
                neurons[i] = new Neuron(biasWeight);
            }
        }
    }

    /**
     * Gets the neurons in this layer.
     *
     * @return The list of neurons in this layer
     */
    Neuron[] getNeurons() {
        return neurons;
    }

    /**
     * Gets the neuron at the specified index in this layer.
     *
     * @param index The index of neuron to retrieve.
     * @return The neuron at specified index.
     */
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
