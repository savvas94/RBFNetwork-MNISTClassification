/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rbfclassificationproject;

import java.util.Arrays;

/**
 *
 * @author Savvas
 */
public class Image {
    
    private int[] data;
    private int label;

    public Image(int[] data, int label) {
        this.data = Arrays.copyOf(data, data.length);
        this.label = label;
    }

    public int[] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }
    
    public int getL2Difference(Image a) {
        int l2 = 0;
        for (int i = 0; i <this. data.length; i++) {
            l2 += Math.pow(this.data[i] - a.data[i],  2);
        }
        //System.out.println("l2:" + l2);
        return l2;
    }
    
    public int getL1Difference(Image a) {
        int l1 = 0;
        for (int i = 0; i <this. data.length; i++) {
            l1 += Math.abs(this.data[i] - a.data[i]);
        }
        //System.out.println("l1:" + l1);
        return l1;
    }
    
    //only compare if the pixels have value 0 or >0.
    public int getL1Difference2(Image a) {
        int diff = 0;
        for (int i = 0; i <this. data.length; i++) {
            if( (this.data[i] == 0 && a.data[i] > 0) || (this.data[i] > 0 && a.data[i] == 0) ){
                diff++;
            }
        }
        return diff;
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 47 * hash + Arrays.hashCode(this.data);
        hash = 47 * hash + this.label;
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Image other = (Image) obj;
        if (this.label != other.label) {
            return false;
        }
        if (!Arrays.equals(this.data, other.data)) {
            return false;
        }
        return true;
    }

    @Override
    public String toString() {
        return "Image{" + "label=" + label + '}';
    }
}
