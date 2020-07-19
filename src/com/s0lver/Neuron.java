package com.s0lver;

public class Neuron {
    private double biasWeight;
    private double cacheBiasWeight;
    private double[] weights;
    private double[] cacheWeights;
    private double gradient;
    private double output;

    static double minWeightValue;
    static double maxWeightValue;


    public Neuron(double biasWeight, double[] weights) {
        this.biasWeight = biasWeight;
        this.cacheBiasWeight = biasWeight;
        this.weights = weights;
        this.cacheWeights = new double[weights.length];
    }

    public Neuron(double biasWeight) {
        this.biasWeight = biasWeight;
        this.cacheBiasWeight = biasWeight;
    }

    /**
     * Constructor used for the bias neurons
     */
    public Neuron() {
        this.weights = null;
        this.gradient = 0;
        this.output = 1;
    }

    public static void setRangeWeight(double minWeightValue, double maxWeightValue) {
        Neuron.minWeightValue = minWeightValue;
        Neuron.maxWeightValue = maxWeightValue;
    }

    /**
     * Updates the weights with the calculated values after the backpropagation step.
     */
    public void updateWeights() {
        // if (this.weights.length - 1 >= 0)
        //     System.arraycopy(this.cacheWeights, 1, this.weights, 1, this.weights.length - 1);
        for (int i = 0; i < this.cacheWeights.length; i++) {
            this.weights[i] = this.cacheWeights[i];
        }
        // this.biasWeight = cacheBiasWeight;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getBiasWeight() {
        return biasWeight;
    }

    public int getNumOfWeights() {
        return weights.length;
    }

    public double getWeight(int index) {
        return weights[index];
    }

    public void setGradient(double gradient) {
        this.gradient = gradient;
    }

    public void setCacheWeight(int index, double value) {
        cacheWeights[index] = value;
    }

    public double getGradient() {
        return gradient;
    }

    /**
     * Sets the weights for a neuron. The specified weights should include the bias weight at the start
     *
     * @param newWeights The new weights to assign
     */
    public void setWeights(double[] newWeights) {
        this.setBiasWeight(newWeights[0]);
        System.arraycopy(newWeights, 1, this.weights, 0, newWeights.length - 1);
    }

    public void setBiasWeight(double biasWeight) {
        this.biasWeight = biasWeight;
    }

    public void setCacheBiasWeight(double cacheBiasWeight) {
        this.cacheBiasWeight = cacheBiasWeight;
    }

    public double[] getWeights() {
        return weights;
    }
}