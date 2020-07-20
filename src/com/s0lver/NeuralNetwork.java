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

        for (int i = 0; i < layerDimensions.size() - 1; i++) {
            int[] dimension = layerDimensions.get(i);
            layers.add(new Layer(dimension[0], dimension[1], true));
        }
        int[] lastDimension = layerDimensions.get(layerDimensions.size() - 1);
        layers.add(new Layer(lastDimension[0], lastDimension[1], false));
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
        feedInputLayer(input);

        // Then forward for the rest of layers
        for (int i = 1; i < layers.size(); i++) {
            final Layer ithLayer = layers.get(i);
            for (int j = 0; j < ithLayer.getNumOfNeurons(); j++) {
                double neuronInput = 0.0;
                final Layer previousLayer = layers.get(i - 1);

                final Neuron biasNeuron = previousLayer.getBiasNeuron();
                neuronInput += biasNeuron.getOutput() * ithLayer.getNeuron(j).getBiasWeight();

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

    private void feedInputLayer(double[] input) {
        var firstLayer = layers.get(0);
        int i = 0;
        for (Neuron neuron : firstLayer.getNeurons()) {
            neuron.setOutput(input[i++]);
        }
    }

    public Prediction predict(TrainingData trainingData) {
        forward(trainingData.getData());
        double[] predicted = new double[outputLayer.getNumOfNeurons()];
        for (int i = 0; i < outputLayer.getNumOfNeurons(); i++) {
            predicted[i] = outputLayer.getNeuron(i).getOutput();
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

    private void calculateLocalGradientForNeuron(Neuron neuron, double causedErrorInNextLayers) {
        double output = neuron.getOutput();
        double outputDerivative = output * (1 - output);
        double delta = causedErrorInNextLayers * outputDerivative;
        neuron.setGradient(delta);
    }

    private void calculateNewWeightsForNeuron(Neuron neuron, Layer previousLayer, double learningRate) {
        double delta = neuron.getGradient();
        double previousOutput = previousLayer.getBiasNeuron().getOutput(); // THIS IS ALWAYS 1!!
        double error = delta * previousOutput;
        double newWeight = neuron.getBiasWeight() - learningRate * error;
        neuron.setCacheBiasWeight(newWeight);

        for (int j = 0; j < neuron.getNumOfWeights(); j++) {
            previousOutput = previousLayer.getNeuron(j).getOutput();
            error = delta * previousOutput;
            newWeight = neuron.getWeight(j) - learningRate * error;
            neuron.setCacheWeight(j, newWeight);
            //                System.out.println(String.format("Layer %s, Neuron %s, weight %s, %s", outIndex, i, j, newWeight));
        }
    }

    private void backward(double learningRate, TrainingData trainingData) {
        int numOfLayers = layers.size();
        int outIndex = numOfLayers - 1;

        // Updating the output layer, for each output neuron
        for (int i = 0; i < outputLayer.getNumOfNeurons(); i++) {
            final Neuron ithNeuron = outputLayer.getNeuron(i);
            double output_i = ithNeuron.getOutput();
            double target_i = trainingData.getExpectedOutput()[i];
            double d_totalError_wrt_out_neuron_i = output_i - target_i;
            calculateLocalGradientForNeuron(ithNeuron, d_totalError_wrt_out_neuron_i);
            calculateNewWeightsForNeuron(ithNeuron, layers.get(outIndex - 1), learningRate);
        }

        // Update subsequent layers
        for (int i = outIndex - 1; i > 0; i--) { // We don't process layer 0 as it is the input layer
            // For all neurons in layer
            final Layer ithLayer = layers.get(i);
            for (int j = 0; j < ithLayer.getNumOfNeurons(); j++) {
                final Neuron jthNeuron = ithLayer.getNeuron(j);
                double output = jthNeuron.getOutput();

                // Las 4 siguientes lineas son para ayudar a los calculos en la capa anterios (sig. ciclo)
                double gradientSum = sumGradient(j, i + 1);
                final double sigmoidDerivative = output * (1 - output);
                double delta = gradientSum * sigmoidDerivative;  // mi error en relacion con lo que causé
                jthNeuron.setGradient(delta);

                double previousOutput = layers.get(i - 1).getBiasNeuron().getOutput(); // THIS IS ALWAYS 1!!
                double error = delta * previousOutput;
                double newWeight = jthNeuron.getBiasWeight() - learningRate * error;
                jthNeuron.setCacheBiasWeight(newWeight);

                // For all its weights
                for (int k = 0; k < jthNeuron.getNumOfWeights(); k++) {
                    // this is actually d_net_jth_neuron wrt input weight k = output neuron k in previous layer
                    double outputNeuronPreviousLayer = layers.get(i - 1).getNeuron(k).getOutput();
                    error = delta * outputNeuronPreviousLayer;
                    newWeight = jthNeuron.getWeight(k) - learningRate * error;
                    jthNeuron.setCacheWeight(k, newWeight);
                    //                    System.out.println(String.format("Layer %s, Neuron %s, weight %s, %s", i, j, k, newWeight));
                }

            }
        }

        // Finally update the inputs (weights) for neurons across all layers (except input layer).
        for (int currentLayer = 1; currentLayer < layers.size(); currentLayer++) {
            Layer layer = layers.get(currentLayer);
            // We don't update the weight of bias neuron (just to reproduce blog results)
            for (int n = 0; n < layer.getNumOfNeurons(); n++) {
                Neuron neuron = layer.getNeuron(n);
                neuron.updateWeights();
            }
        }
    }

    public double sumGradient(int neuron, int layer) { // recoger la culpa que causó  neuron (de la capa layer-1) en la capa layer
        double gradientSum = 0;
        Layer currentLayer = layers.get(layer);
        // we start at 1  because we don't calculate the gradient of the bias neurons, we skip them.
        for (int i = 0; i < currentLayer.getNumOfNeurons(); i++) {
            Neuron currentNeuron = currentLayer.getNeuron(i);

            final double localGradient = currentNeuron.getGradient();
            final double weight = currentNeuron.getWeight(neuron);
            final double d_grad_currentNeuron_wrt_neuron = weight * localGradient;
            gradientSum += d_grad_currentNeuron_wrt_neuron;
        }
        return gradientSum;
    }

    public LinkedList<Layer> getLayers() {
        return layers;
    }
}
