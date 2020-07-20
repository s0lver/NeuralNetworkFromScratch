package com.s0lver.neuralnetwork;

import java.util.LinkedList;
import java.util.List;

import static com.s0lver.neuralnetwork.Utils.calculateSigmoidDerivative;

/**
 * Models a neural network, with layers that can define a bias neuron.
 */
public class NeuralNetwork {
    /**
     * The layers of the neural network.
     */
    private final LinkedList<Layer> layers;

    /**
     * A reference to the output layer
     */
    private final Layer outputLayer;

    /**
     * The learning rate used during training.
     */
    private double learningRate;

    /**
     * Constructor
     *
     * @param layerDimensions The specification of the dimensions for the layers.
     *                        The input size for input layer should be set to 0.
     */
    public NeuralNetwork(List<int[]> layerDimensions) {
        this.layers = new LinkedList<>();

        for (int i = 0; i < layerDimensions.size() - 1; i++) {
            int[] dimension = layerDimensions.get(i);
            layers.add(new Layer(dimension[0], dimension[1], true));
        }
        int[] lastDimension = layerDimensions.get(layerDimensions.size() - 1);
        layers.add(new Layer(lastDimension[0], lastDimension[1], false));

        this.outputLayer = layers.getLast();
    }

    /**
     * Updates (directly, from the exterior) the weights of the neuron at the specified position in the specified layer.
     *
     * @param layerIndex  The index of the layer where the neuron is.
     * @param neuronIndex The index of the neuron.
     * @param biasWeight  The bias weight to assign
     * @param weights     The input weights to assign for the neuron.
     */
    public void setWeightsForNeuronInLayer(int layerIndex, int neuronIndex, double biasWeight, double[] weights) {
        layers.get(layerIndex).getNeuron(neuronIndex).setWeights(biasWeight, weights);
    }

    /**
     * Performs the forward step for the network using the specified input data and current network's configuration (weights).
     *
     * @param input An input (n-dimensional vector) for the network.
     */
    private void forward(double[] input) {
        feedInputLayer(input);

        // Then forward for the rest of layers
        for (int i = 1; i < layers.size(); i++) {
            final Layer ithLayer = layers.get(i);
            for (Neuron neuron : ithLayer.getNeurons()) {
                double neuronInput = 0.0;
                final Layer previousLayer = layers.get(i - 1);

                final Neuron biasNeuron = previousLayer.getBiasNeuron();
                neuronInput += biasNeuron.getOutput() * neuron.getBiasWeight();

                for (int k = 0; k < previousLayer.getNumOfNeurons(); k++) {
                    final double product = previousLayer.getNeuron(k).getOutput() * neuron.getWeight(k);
                    neuronInput += product;
                }

                final double output = Utils.Sigmoid(neuronInput);
                neuron.setOutput(output);
                // System.out.println(String.format("layer%s * layer%s[%s] = %s. sig(%s) = %s", (i - 1), i, j, neuronInput, neuronInput, out));
            }
        }
    }

    /**
     * Feeds the input data to the neuron's input layer.
     *
     * @param input The data to feed.
     */
    private void feedInputLayer(double[] input) {
        int i = 0;
        Layer firstLayer = layers.get(0);
        for (Neuron neuron : firstLayer.getNeurons()) {
            neuron.setOutput(input[i++]);
        }
    }

    /**
     * Generates (predicts) the output of the network using the input data.
     *
     * @param trainingData The sample data to use.
     * @return A Prediction object, including the outcome and error by the network.
     */
    public Prediction predict(TrainingData trainingData) {
        forward(trainingData.getData());
        double[] predicted = new double[outputLayer.getNumOfNeurons()];
        for (int i = 0; i < outputLayer.getNumOfNeurons(); i++) {
            predicted[i] = outputLayer.getNeuron(i).getOutput();
        }
        return new Prediction(predicted, trainingData.getExpectedOutput());
    }

    /**
     * Performs the training of the network.
     *
     * @param trainingDataset    The training dataset to use.
     * @param trainingIterations The number of iterations to perform.
     * @param learningRate       The learning rate for updating weights.
     */
    public void train(TrainingData[] trainingDataset, int trainingIterations, double learningRate) {
        this.learningRate = learningRate;
        for (int i = 0; i < trainingIterations; i++) {
            for (TrainingData trainingData : trainingDataset) {
                forward(trainingData.getData());
                backward(trainingData);
            }
        }
    }

    /**
     * Calculates the gradient (error observed) for the specified neuron.
     *
     * @param neuron                    The neuron whose gradient will be calculated.
     * @param generatedErrorInNextLayer The error caused by the neuron in the next layer (i.e., the sum of the gradient
     *                                  for the neurons of the next layer connected to this neuron). When the neuron is
     *                                  on the output layer, this value refers to the difference between network's
     *                                  output and expected target value.
     */
    private void calculateLocalGradientForNeuron(Neuron neuron, double generatedErrorInNextLayer) {
        double output = neuron.getOutput();
        double outputDerivative = calculateSigmoidDerivative(output);
        double delta = generatedErrorInNextLayer * outputDerivative;
        neuron.setGradient(delta);
    }

    /**
     * Calculates the new weights for the specified neuron.
     *
     * @param neuron        The neuron whose new weights will be calculated.
     * @param previousLayer The previous layer, that is the layer whose neurons are feeding the specified neuron.
     * @param learningRate  The learning rate to use fr updating weights.
     */
    private void calculateNewWeightsForNeuron(Neuron neuron, Layer previousLayer, double learningRate) {
        double delta = neuron.getGradient();
        double previousOutput = previousLayer.getBiasNeuron().getOutput(); // yes, this is always 1 for the bias.
        double error = delta * previousOutput;
        double newWeight = neuron.getBiasWeight() - learningRate * error;
        neuron.setCacheBiasWeight(newWeight);

        for (int j = 0; j < neuron.getNumOfWeights(); j++) {
            previousOutput = previousLayer.getNeuron(j).getOutput();
            error = delta * previousOutput;
            newWeight = neuron.getWeight(j) - learningRate * error;
            neuron.setCacheWeight(j, newWeight);
            // System.out.println(String.format("Layer %s, Neuron %s, weight %s, %s", outIndex, i, j, newWeight));
        }
    }

    /**
     * Performs the backwards stage of the neural network's training.
     *
     * @param trainingData The training data to use.
     */
    private void backward(TrainingData trainingData) {
        int numOfLayers = layers.size();
        int outIndex = numOfLayers - 1;
        Layer previousLayer = layers.get(outIndex - 1);

        // Updating the output layer, for each output neuron
        for (int i = 0; i < outputLayer.getNumOfNeurons(); i++) {
            final Neuron ithNeuron = outputLayer.getNeuron(i);
            double output_i = ithNeuron.getOutput();
            double target_i = trainingData.getExpectedOutput()[i];
            double d_totalError_wrt_out_neuron_i = output_i - target_i;

            calculateLocalGradientForNeuron(ithNeuron, d_totalError_wrt_out_neuron_i);
            calculateNewWeightsForNeuron(ithNeuron, previousLayer, learningRate);
        }

        // Update subsequent layers
        for (int i = outIndex - 1; i > 0; i--) {
            final Layer ithLayer = layers.get(i);
            previousLayer = layers.get(i - 1);
            for (int j = 0; j < ithLayer.getNumOfNeurons(); j++) {
                final Neuron jthNeuron = ithLayer.getNeuron(j);
                double gradientSum = sumGradient(j, i + 1);

                calculateLocalGradientForNeuron(jthNeuron, gradientSum);
                calculateNewWeightsForNeuron(jthNeuron, previousLayer, learningRate);
            }
        }
        updateWeights();
    }

    /**
     * Update the weights for the neurons in the network's layers using the new weights calculated in the backpropagation stage.
     */
    private void updateWeights() {
        for (int currentLayer = 1; currentLayer < layers.size(); currentLayer++) {
            Layer layer = layers.get(currentLayer);
            for (int n = 0; n < layer.getNumOfNeurons(); n++) {
                Neuron neuron = layer.getNeuron(n);
                neuron.updateWeights();
            }
        }
    }

    /**
     * Calculates the error (gradient) caused by this neuron in the neurons of the next layer (layer).
     *
     * @param neuron The neuron whose gradient will be calculated.
     * @param layer  The layer next to the specified neuron.
     * @return The sum of the gradients for the neurons next layer that are connected to the specified neuron.
     */
    public double sumGradient(int neuron, int layer) {
        double gradientSum = 0;
        Layer currentLayer = layers.get(layer);
        for (int i = 0; i < currentLayer.getNumOfNeurons(); i++) {
            Neuron currentNeuron = currentLayer.getNeuron(i);

            final double localGradient = currentNeuron.getGradient();
            final double weight = currentNeuron.getWeight(neuron);
            final double d_grad_currentNeuron_wrt_neuron = weight * localGradient;
            gradientSum += d_grad_currentNeuron_wrt_neuron;
        }
        return gradientSum;
    }

    /**
     * Util method that gets the layers in the network.
     *
     * @return The list of layers in this network.
     */
    public LinkedList<Layer> getLayers() {
        return layers;
    }
}
