/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author yang
 */

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class conglemerative {
    
    public static int classIndex;
    public static ArrayList<ArrayList<Integer>> multiArrayList = new ArrayList<>();
    public static ArrayList<Double> multiAccuracyList = new ArrayList<>();
    public static int count_no=0, count_yes=0, count_multi=0;
    public static double noArrayAccuracy;
    public static double yesArrayAccuracy;
    
    // preprocess the dataset, divide the participants into 3 clusters (ALL YES, ALL NO, and multi-node)
    public static void preprocessing() throws Exception{
        // to store user clusters
        ArrayList<Integer> noArray = new ArrayList<>();
        ArrayList<Integer> yesArray = new ArrayList<>();
        ArrayList<Integer> multiArray = new ArrayList<>();
        // ALL NO and ALL YES
        String t1="N0 [label=\"0";
        String t2="N0 [label=\"1";  
        //String t3="N0 [label=\"cstore";
        
        DataSource source = new DataSource("docs/samsung.arff");
        Instances allusers=source.getDataSet();
        classIndex=allusers.numAttributes()-1;
        allusers.setClassIndex(classIndex);
        //System.out.println(allusers.numInstances());

        for (int i = 0; i < allusers.numInstances(); i = i + 12) {
            int userID = (int)allusers.instance(i).value(0);
            Instances singleUserInstances = new Instances(allusers, i, 12);
            FilteredClassifier cls = wekaFunctions.train(singleUserInstances, classIndex); // train
//            double accuracy =  wekaFunctions.eval(cls, singleUserInstances,singleUserInstances); // eval
//            System.out.println("User #:" +userID);
//            System.out.println("Classifier :" +fc);
//            System.out.println("Accuracy :" +accuracy);

            //cls.graph() store this in a string and use string functions to parse it; put in if conditions to determine the clusters
            if(cls.graph().contains(t1)){
                count_no++;
                noArray.add(userID);
            }
            else if(cls.graph().contains(t2)){
                count_yes++;
                yesArray.add(userID);
            }
            else {
                count_multi++;
                multiArray.add(userID);
                ArrayList<Integer> currentUser = new ArrayList<>();
                currentUser.add(userID);
                multiArrayList.add(currentUser);
                multiAccuracyList.add(wekaFunctions.eval(cls, singleUserInstances,singleUserInstances));
                //NoneSingleNodeUserAccuracyList[array3Index++][0] = (float)accuracy;
            } 
        }
        System.out.println("Single 'NO' node trees: " + count_no);
        System.out.println("Single 'YES' node trees: " + count_yes);
        System.out.println("Multi-node trees: " + count_multi);        
//        System.out.println("multiArray's length is: " + multiArray.size());
//        for(Integer temp:multiArray){  
//            System.out.println(temp);
//        } 

        arffFunctions.generateArff(noArray, "docs/samsung_header.txt", "bottomupNoArray.arff");
        arffFunctions.generateArff(yesArray, "docs/samsung_header.txt", "bottomupYesArray.arff");  
        arffFunctions.generateArff(multiArray, "docs/samsung_header.txt", "bottomupMultiArray.arff");
        
        DataSource sourceNoArray = new DataSource("docs/bottomupNoArray.arff");
        DataSource sourceYesArray = new DataSource("docs/bottomupYesArray.arff");
        DataSource sourceMultiArray = new DataSource("docs/bottomupMultiArray.arff");
        
        Instances dataNoArray = sourceNoArray.getDataSet();
        Instances dataYesArray = sourceYesArray.getDataSet();
        Instances dataMultiArray = sourceMultiArray.getDataSet();
        
	dataNoArray.setClassIndex(classIndex);
	dataYesArray.setClassIndex(classIndex);
        dataMultiArray.setClassIndex(classIndex);
        
        noArrayAccuracy = wekaFunctions.trainAndEval(dataNoArray, dataNoArray, classIndex);
        yesArrayAccuracy = wekaFunctions.trainAndEval(dataYesArray, dataYesArray, classIndex);
        double multiArrayAccuracy = wekaFunctions.trainAndEval(dataMultiArray, dataMultiArray, classIndex);
        System.out.println("noArray's accuracy is: " + noArrayAccuracy);
        System.out.println("yesArray's accuracy is: " + yesArrayAccuracy);
        System.out.println("multiArray's accuracy is: " + multiArrayAccuracy);
        System.out.println("Initial 3 Clusters Overall Accuracy is: " + (noArrayAccuracy*count_no+yesArrayAccuracy*count_yes+multiArrayAccuracy*count_multi)/(count_no+count_yes+count_multi));
        //System.out.println(wekaFunctions.train(dataNoArray, classIndex));
        //System.out.println(wekaFunctions.train(dataYesArray, classIndex));
        //System.out.println(wekaFunctions.train(dataMultiArray, classIndex));
    }

// Merge part of conglemerative, we deal with multiArray here
    public static void merge(int ROUNDS) throws Exception{
        
        double maxPairAccuracy = 0;
        int maxIndexLeftHand = 0;
        int maxIndexRightHand = 0;
        int index=1;
        
        for (int round = 0; round < ROUNDS; round++) {            
            System.out.println("Round: " + (round+1) );
            System.out.println(multiArrayList.size());
            System.out.println(multiAccuracyList.size());
            
            for (int i = 0; i < multiArrayList.size(); i++) {
                for (int j = i+1; j < multiArrayList.size(); j++) {
//            int i = 422;
//            int j = 423;
                    ArrayList<Integer> currentPair = new ArrayList<>(multiArrayList.get(i));
                    currentPair.addAll(multiArrayList.get(j));
                    
                    arffFunctions.generateArff(currentPair, "docs/samsung_header.txt", "currentPair.arff");
                    DataSource sourceCurrentPair = new DataSource("docs/currentPair.arff");
                    Instances dataCurrentPair = sourceCurrentPair.getDataSet();
                    double currentPairAccuracy = wekaFunctions.trainAndEval
                        (dataCurrentPair, dataCurrentPair, classIndex);
//                    System.out.print((index++) + ": ");
//                    for(Integer temp:multiArrayList.get(i)){  
//                        System.out.print(temp);
//                        System.out.print(" ");
//                    }
//                    System.out.print(",");
//                    for(Integer temp:multiArrayList.get(j)){  
//                        System.out.print(temp);
//                        System.out.print(" ");
//                    }
//                    System.out.println(":" + currentPairAccuracy);
                    if (currentPairAccuracy == 100.0) {
                        maxPairAccuracy = currentPairAccuracy;
                        maxIndexLeftHand = i;
                        maxIndexRightHand = j;
                        break;
                    }
                    else if (currentPairAccuracy > maxPairAccuracy) {
                        maxPairAccuracy = currentPairAccuracy;
                        maxIndexLeftHand = i;
                        maxIndexRightHand = j;
                    }
                }
                if (maxPairAccuracy == 100.0) {
                    break;                                                    
                }
            }

            System.out.println(maxIndexLeftHand + "," + maxIndexRightHand + ": " + maxPairAccuracy);
            ArrayList<Integer> leftHand = multiArrayList.get(maxIndexLeftHand);
            //System.out.println("LefthandAcc: " + multiAccuracyList.get(maxIndexLeftHand));
            leftHand.addAll(multiArrayList.get(maxIndexRightHand));
            multiAccuracyList.set(maxIndexLeftHand, maxPairAccuracy);
            //System.out.println("LefthandAcc: " + multiAccuracyList.get(maxIndexLeftHand));
            //System.out.println("Lefthand: " + leftHand);
            multiArrayList.remove(maxIndexRightHand);
            multiAccuracyList.remove(maxIndexRightHand);
            
            System.out.println("---------------------------------------");
            System.out.println("ROUNDS end: ");
            for (ArrayList<Integer> next : multiArrayList) {
                printRow(next);
            }
            System.out.println("======================================");
            maxPairAccuracy = 0;
            maxIndexLeftHand = 0;
            maxIndexRightHand = 0;
            index = 1;
        }       
    }
    
    public static void printRow(ArrayList<Integer> row){
        for(Integer temp:row){  
            System.out.print(temp);
            System.out.print(",");
        }
        System.out.println();
    }
    
    public static void printRow(int[] row) {
        for (int i : row) {
            System.out.print(i);
            System.out.print("\t");
        }
        System.out.println();
    }

    public static void printRow(float[] row) {
        for (float i : row) {
            System.out.print(i);
            System.out.print("\t");
        }
        System.out.println();
    }
    
    public static void main(String[] args) throws Exception{
        preprocessing();
        //merge(434);
        test();
    }
    
    public static void test() throws IOException, Exception{
        
        int classIndex = 6;
        
        int l1 = constantVar.cluster1.length;
        int l2 = constantVar.cluster2.length;
        int l3 = constantVar.cluster3.length;
        int l4 = constantVar.cluster4.length;
        int l5 = constantVar.cluster5.length;
        
        arffFunctions.generateArff(constantVar.cluster1, "docs/samsung_header.txt", "model1.arff");
        arffFunctions.generateArff(constantVar.cluster2, "docs/samsung_header.txt", "model2.arff");
        arffFunctions.generateArff(constantVar.cluster3, "docs/samsung_header.txt", "model3.arff");
        arffFunctions.generateArff(constantVar.cluster4, "docs/samsung_header.txt", "model4.arff");
        arffFunctions.generateArff(constantVar.cluster5, "docs/samsung_header.txt", "model5.arff");

        DataSource source1 = new DataSource("docs/model1.arff");
        DataSource source2 = new DataSource("docs/model2.arff");
        DataSource source3 = new DataSource("docs/model3.arff");
        DataSource source4 = new DataSource("docs/model4.arff");
        DataSource source5 = new DataSource("docs/model5.arff");

        Instances data1 = source1.getDataSet();
        Instances data2 = source2.getDataSet();
        Instances data3 = source3.getDataSet();
        Instances data4 = source4.getDataSet();
        Instances data5 = source5.getDataSet();

        data1.setClassIndex(classIndex);
        data2.setClassIndex(classIndex);
        data3.setClassIndex(classIndex);
        data4.setClassIndex(classIndex);
        data5.setClassIndex(classIndex);

        FilteredClassifier fc1 = wekaFunctions.train(data1, classIndex);
        FilteredClassifier fc2 = wekaFunctions.train(data2, classIndex);
        FilteredClassifier fc3 = wekaFunctions.train(data3, classIndex);
        FilteredClassifier fc4 = wekaFunctions.train(data4, classIndex);
        FilteredClassifier fc5 = wekaFunctions.train(data5, classIndex);

        double Array1Accuracy = wekaFunctions.eval(fc1, data1, data1);
        double Array2Accuracy = wekaFunctions.eval(fc2, data2, data2);
        double Array3Accuracy = wekaFunctions.eval(fc3, data3, data3);
        double Array4Accuracy = wekaFunctions.eval(fc4, data4, data4);
        double Array5Accuracy = wekaFunctions.eval(fc5, data5, data5);
        
        System.out.println("=============================================================================");
        System.out.println("fc1:\n " + fc1);
        System.out.println("fc2:\n " + fc2);
        System.out.println("fc3:\n " + fc3);
        System.out.println("fc4:\n " + fc4);
        System.out.println("fc5:\n " + fc5);
        
        System.out.println("Array1's accuracy: " + Array1Accuracy);
        System.out.println("Array2's accuracy: " + Array2Accuracy);
        System.out.println("Array3's accuracy: " + Array3Accuracy);
        System.out.println("Array4's accuracy: " + Array4Accuracy);
        System.out.println("Array5's accuracy: " + Array5Accuracy);
        
//        System.out.println(l1+l2+count_no+count_yes + " participants' Average Accuracy: "
//                + (Array1Accuracy*l1
//                        +Array2Accuracy*l2
//                        +noArrayAccuracy*count_no
//                        +yesArrayAccuracy*count_yes)
//                        /(l1+l2+count_no+count_yes)
//        );
        System.out.println(l1+l2+l3+l4+l5+count_no+count_yes + " participants' Average Accuracy: "
                + (Array1Accuracy*l1
                        +Array2Accuracy*l2
                        +Array3Accuracy*l3
                        +Array4Accuracy*l4
                        +Array5Accuracy*l5
                        +noArrayAccuracy*count_no
                        +yesArrayAccuracy*count_yes)
                        /(l1+l2+l3+l4+l5+count_no+count_yes)
        );
    }
}    



