package com.s0lver.neuralnetwork;

/**
 * Models a neuron with input weights (and independent bias weight).
 * For easing calculations, it defines a cache of weights that are used when updating the weights after a backpropagation step.
 */
public class Neuron {
    /**
     * The input bias weight
     */
    private double biasWeight;

    /**
     * A cache for the bias weight.
     */
    private double cacheBiasWeight;

    /**
     * The input weights for the neuron. I.e., weights that come from neurons in previous layer.
     */
    private double[] weights;

    /**
     * A cache for weights.
     */
    private double[] cacheWeights;

    /**
     * The local gradient (error) observed by the neuron.
     */
    private double gradient;

    /**
     * The output by the network. This is the result of activationFunction(input).
     */
    private double output;

    /**
     * Util field to limit the minimum value for neuron's weight
     */
    static double minWeightValue;

    /**
     * Util field to limit the maximum value for neuron's weight
     */
    static double maxWeightValue;

    /**
     * Constructor for neurons in inner layers (except for bias neurons).
     *
     * @param biasWeight The bias weight to assign to the neuron.
     * @param weights    The list of weights to assign to the neuron.
     */
    public Neuron(double biasWeight, double[] weights) {
        this.biasWeight = biasWeight;
        this.cacheBiasWeight = biasWeight;
        this.weights = weights;
        this.cacheWeights = new double[weights.length];
    }

    /**
     * Constructor for neurons in input layer.
     *
     * @param biasWeight The bias weight to assign to the neuron.
     */
    public Neuron(double biasWeight) {
        this.biasWeight = biasWeight;
        this.cacheBiasWeight = biasWeight;
    }

    /**
     * Constructor used for bias neurons.
     */
    public Neuron() {
        this.weights = null;
        this.gradient = 0;
        this.output = 1;
    }

    /**
     * Util method to adjust the range for neuron's weights.
     *
     * @param minWeightValue The minimum value for weights.
     * @param maxWeightValue The maximum value for weights.
     */
    public static void setRangeWeight(double minWeightValue, double maxWeightValue) {
        Neuron.minWeightValue = minWeightValue;
        Neuron.maxWeightValue = maxWeightValue;
    }

    /**
     * Updates the weights with the calculated values after the backpropagation step.
     */
    public void updateWeights() {
        System.arraycopy(this.cacheWeights, 0, this.weights, 0, this.cacheWeights.length);
        this.biasWeight = cacheBiasWeight;
    }

    /**
     * Gets neuron's output.
     *
     * @return The neuron's output.
     */
    public double getOutput() {
        return output;
    }

    /**
     * Sets the output for the neuron.
     *
     * @param output The output to assign
     */
    public void setOutput(double output) {
        this.output = output;
    }

    /**
     * Gets the value for the bias weight.
     *
     * @return The bias weight for this neuron
     */
    public double getBiasWeight() {
        return biasWeight;
    }

    /**
     * Gets the number of weights for the neuron (does not include the bias weight).
     *
     * @return The number of weights for the neuron.
     */
    public int getNumOfWeights() {
        return weights.length;
    }

    /**
     * Gets the weight at the specified index.
     *
     * @param index The index of the weight to retrieve.
     * @return The weight at the specified index.
     */
    public double getWeight(int index) {
        return weights[index];
    }

    /**
     * Sets the neuron's gradient.
     *
     * @param gradient The gradient value to assign.
     */
    public void setGradient(double gradient) {
        this.gradient = gradient;
    }

    /**
     * Gets the neuron's gradient.
     *
     * @return The neuron's gradient.
     */
    public double getGradient() {
        return gradient;
    }

    /**
     * Sets the cache weight at the specified index for this neurons.
     *
     * @param index The index of the cache weight to update.
     * @param value The new value to assign.
     */
    public void setCacheWeight(int index, double value) {
        cacheWeights[index] = value;
    }

    /**
     * Sets the weights for a neuron.
     *
     * @param biasWeight The new bias weight to assign.
     * @param newWeights The new weights to assign
     */
    public void setWeights(double biasWeight, double[] newWeights) {
        this.biasWeight = biasWeight;
        System.arraycopy(newWeights, 0, this.weights, 0, newWeights.length);
    }

    /**
     * Sets the cache bias weight for the neuron.
     *
     * @param cacheBiasWeight The new value to assign.
     */
    public void setCacheBiasWeight(double cacheBiasWeight) {
        this.cacheBiasWeight = cacheBiasWeight;
    }

    public double[] getWeights() {
        return weights;
    }
}