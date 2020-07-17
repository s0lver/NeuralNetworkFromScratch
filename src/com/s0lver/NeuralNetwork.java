package com.s0lver;

import java.util.LinkedList;
import java.util.List;

import static com.s0lver.Utils.calculateError;

public class NeuralNetwork {
    private final LinkedList<Layer> layers;

    private final Layer outputLayer;

    static {
        Neuron.setRangeWeight(0.0, 1.0);
    }

    public NeuralNetwork(List<int[]> layerDimensions) {
        this.layers = new LinkedList<>();
        for (int[] dim : layerDimensions) {
            layers.add(new Layer(dim[0] + 1, dim[1] + 1));
        }
        this.outputLayer = layers.getLast();
    }

    public void setWeightsForNeuronInLayer(int layer, int neuron, double[] weights) {
        layers.get(layer).getNeuron(neuron).setWeights(weights);
    }

    /**
     * Performs the forward (or predict for that matter) for the network using the input data.
     *
     * @param input An input (n-dimensional vector) for the network.
     */
    public void forward(double[] input) {
        var firstLayer = layers.get(0);
        Neuron[] neurons = firstLayer.getNeurons();
        for (int i = 1; i < neurons.length; i++) {
            Neuron neuron = neurons[i];
            neuron.setOutput(input[i - 1]);
        }

        // Then forward for the rest of layers
        for (int i = 1; i < layers.size(); i++) {
            final Layer ithLayer = layers.get(i);
            // for (int j = 0; j < ithLayer.getNumOfNeurons(); j++) {
            for (int j = 1; j < ithLayer.getNumOfNeurons(); j++) {
                double neuronInput = 0.0;
                final Layer previousLayer = layers.get(i - 1);
                for (int k = 0; k < previousLayer.getNumOfNeurons(); k++) {
                    final double product = previousLayer.getNeuron(k).getOutput() * ithLayer.getNeuron(j).getWeight(k);
                    neuronInput += product;
                }
                final double out = Utils.Sigmoid(neuronInput);
                ithLayer.getNeuron(j).setOutput(out);
                // System.out.println(String.format("layer%s * layer%s[%s] = %s. sig(%s) = %s", (i - 1), i, j, neuronInput, neuronInput, out));
            }
        }
    }

    public Prediction predict(TrainingData trainingData) {
        forward(trainingData.getData());
        double[] predicted = new double[outputLayer.getNumOfNeurons() - 1];
        for (int i = 1; i < outputLayer.getNumOfNeurons(); i++) {
            predicted[i - 1] = outputLayer.getNeuron(i).getOutput();
        }
        double error = calculateError(predicted, trainingData.getExpectedOutput());
        return new Prediction(predicted, trainingData.getExpectedOutput(), error);
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
        int numOfLayers = layers.size();
        int outIndex = numOfLayers - 1;

        // Updating the output layer, for each output neuron
        for (int i = 1; i < outputLayer.getNumOfNeurons(); i++) {
            final Neuron ithNeuron = outputLayer.getNeuron(i);
            double output_i = ithNeuron.getOutput();
            double target_i = trainingData.getExpectedOutput()[i - 1]; // i-1 because the output does not have the bias...
            double d_totalError_wrt_out_neuron_i = output_i - target_i;
            final double sigmoidDerivative = output_i * (1 - output_i);
            double delta = d_totalError_wrt_out_neuron_i * sigmoidDerivative;
            ithNeuron.setGradient(delta);
            for (int j = 0; j < ithNeuron.getNumOfWeights(); j++) {
                double previousOutput = layers.get(outIndex - 1).getNeuron(j).getOutput();
                double error = delta * previousOutput;
                final double newWeight = ithNeuron.getWeight(j) - learningRate * error;
                ithNeuron.setCacheWeight(j, newWeight);
                // System.out.println(String.format("Layer %s, Neuron %s, weight %s, %s", outIndex, i, j, newWeight));
            }
        }

        // Update subsequent layers
        for (int i = outIndex - 1; i > 0; i--) { // We don't process layer 0 as it is the input layer
            // For all neurons in layer
            final Layer ithLayer = layers.get(i);
            //             for (int j = 0; j < ithLayer.getNumOfNeurons(); j++) {
            for (int j = 1; j < ithLayer.getNumOfNeurons(); j++) {
                final Neuron jthNeuron = ithLayer.getNeuron(j);
                double output = jthNeuron.getOutput();
                double gradientSum = sumGradient(j, i + 1);
                double delta = (gradientSum) * (output * (1 - output));
                jthNeuron.setGradient(delta);

                // For all its weights
                for (int k = 0; k < jthNeuron.getNumOfWeights(); k++) {
                    double previousOutput = layers.get(i - 1).getNeuron(k).getOutput();
                    double error = delta * previousOutput;
                    final double newWeight = jthNeuron.getWeight(k) - learningRate * error;
                    jthNeuron.setCacheWeight(k, newWeight);
                    // System.out.println(String.format("Layer %s, Neuron %s, weight %s, %s", i, j, k, newWeight));
                }
            }
        }

        // Finally update the inputs (weights) for neurons across all layers (except input layer).
        for (int currentLayer = 1; currentLayer < layers.size(); currentLayer++) {
            Layer layer = layers.get(currentLayer);
            // We don't update the weight of bias neuron (just to reproduce blog results)
            for (int n = 1; n < layer.getNumOfNeurons(); n++) {
                Neuron neuron = layer.getNeuron(n);
                neuron.updateWeights();
            }
        }
    }

    public double sumGradient(int neuron, int layer) {
        double gradientSum = 0;
        Layer currentLayer = layers.get(layer);
        // we start at 1  because we don't calculate the gradient of the bias neurons, we skip them.
        for (int i = 1; i < currentLayer.getNumOfNeurons(); i++) {
            Neuron currentNeuron = currentLayer.getNeuron(i);
            gradientSum += currentNeuron.getWeight(neuron) * currentNeuron.getGradient();
        }
        return gradientSum;
    }
}
