/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rbfclassificationproject;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Implements an rbf neuron.
 * 
 * @author Savvas
 */
public class RBFNeuron implements Serializable{
    
    private final List<Double> center;
    private final double beta;

    /**
     * Creates an rbf neuron with the given parameters
     * 
     * @param center the center of the neuron
     * @param beta the beta of the gaussian function.
     */
    public RBFNeuron(List<Double> center, double beta) {
        this.center = new ArrayList<>(center);
        this.beta = beta;
    }

    /**
     * Calculates the value of the neuron for a given input vector. The function
     * used is the gaussian function multiplied with 100, but with max value 1,
     * so that all neurons have values [0, 1]. The multiplication is needed because
     * otherwise the results are always too small for the back propagation algorithm
     * and the training too slow.
     * 
     * @param input the input vector
     * @return the result of the gaussian function of the distance of the input from the vector.
     */
    public double output(List<Double> input) {
        return Math.min(100*Math.pow(Math.E, - (beta * squareDistance(input))), 1);
    }
    
    private double squareDistance(List<Double> input) {
        double sum = 0;
        for (int i = 0; i < center.size(); i++) {
            sum += Math.pow(center.get(i) - input.get(i), 2);
        }
        return Math.sqrt(sum);
    }
    
}
