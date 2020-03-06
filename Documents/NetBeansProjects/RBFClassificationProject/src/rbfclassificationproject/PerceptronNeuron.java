/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rbfclassificationproject;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 *
 * @author Savvas
 */
public class PerceptronNeuron implements Serializable{
    
    private List<Double> weights;
    private double dotProduct;
    private double y;
    
    private static final Random rand = new Random();
    private static final double a = 1;
    private static final double eta = 0.5;

    /**
     *
     * @param inputSize the size of the vector that will be given to this neuron as input.
     */
    public PerceptronNeuron(int inputSize) {
        weights = new ArrayList<>(inputSize);
        for (int i = 0; i < inputSize; i++) {
            weights.add((double)rand.nextFloat() - 0.5);
        }
        weights.add((double)rand.nextFloat() - 0.5);
        dotProduct = 0;
        y = 0;
    }
    
    public double output(List<Double> input) {
        //when the output of the neuron is calculated, store the dot product and
        //the sigm so that the back propagation does not need to do it again.        
        this.dotProduct = dotProduct(input);
        this.y = sigm(this.dotProduct);
        return this.y;
    }
    
    public double adjustWeights(List<Double> input, double target) {
        double phiDer = sigmDer(this.dotProduct); //when we are in the ajdust weights phase, the dot product and the y are already calculated and stored
        double error = target-this.y;
        
        for (int i = 0; i < weights.size()-1; i++) {
            double weight = weights.get(i);
            double change =  eta*error*phiDer*input.get(i);
            weights.set(i, weight + change); // wji = wji + eta*ej*f'(uj)*yi
        }
        //do this for the bias. The weight of the bias is in the last position of the weights
        double weight = weights.get(weights.size()-1);
        weights.set(weights.size()-1, weight + eta*error*phiDer); // wji = wji + eta*ej*f'(uj)*yi -> the bias is 1, so yi is 1.
        return error*phiDer; //return neuron's delta
    }
    
    public double adjustWeightsHidden(List<Double> input, double error){
        double phiDer = sigmDer(this.dotProduct); //when we are in the ajdust weights phase, the dot product is already calculated and stored
        
        for (int i = 0; i < weights.size()-1; i++) {
            double weight = weights.get(i);
            weights.set(i, weight + eta*error*phiDer*input.get(i)); // wji = wji + eta*ej*f'(uj)*yi
        }
        //do this for the bias. The weight of the bias is in the last position of the weights
        double weight = weights.get(weights.size()-1);
        weights.set(weights.size()-1, weight + eta*error*phiDer); // wji = wji + eta*ej*f'(uj)*yi -> the bias is 1, so yi is 1.
        return error*phiDer; //return neuron's delta
    };
    
    public double getWeight(int index) {
        return weights.get(index);
    }
    
    private double dotProduct(List<Double> input) {
        double sum = 0;
        for (int i = 0; i < weights.size()-1; i++) {
            sum += input.get(i)*weights.get(i);            
        }
        sum += weights.get(weights.size()-1); //add the bias
        return sum;
    }
    
    private double sigm(double x) {
        return 1 / (1 + Math.pow(Math.E, -a*x));
    }
    
    private double sigmDer(double x) {
        double sigmRes = this.y;
        return a * ( 1-sigmRes) * sigmRes;
    }
}