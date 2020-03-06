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
 * Implements a layer of rbf neurons.
 * 
 * @author Savvas
 */
public class RBFLayer implements Serializable{
    
    private List<RBFNeuron> layer;

    public RBFLayer(List<RBFNeuron> centers) {
        layer = new ArrayList<>(centers);
    }
    
    /**
     * Calculates the output of each neuron of the layer for a given input vector.
     * 
     * @param input the input vector
     * @return a list with the results of the neurons
     */
    public List<Double> output(List<Double> input) {
        List<Double> layerOutput = new ArrayList<>(layer.size());
        for (RBFNeuron neuron : layer) {
            layerOutput.add(neuron.output(input));
        }
        
        return layerOutput;
    }
    
    public int getLayerSize() {
        return this.layer.size();
    }
    
}
