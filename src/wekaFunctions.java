import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

public class wekaFunctions {
    
        public static FilteredClassifier getFilteredClassifier()
        {
            Remove rm = new Remove();
            rm.setAttributeIndices("1");  // REMOVING ID ATTRIBUTE AS THAT WON'T BE INPUT TO THE CLASSIFIER
            //rm.setAttributeIndices("1");
            // classifier
            J48 j48 = new J48();
            //j48.setUnpruned(true);        // using an unpruned J48
            // meta-classifier
            FilteredClassifier cls = new FilteredClassifier();
            cls.setFilter(rm);
            cls.setClassifier(j48);
            return cls;
        }
        
        public static FilteredClassifier train(ArrayList<Integer> trainArray) throws Exception
	{
            arffFunctions.generateArff(trainArray, "docs/samsung_header.txt", "modelTrain.arff");
            DataSource sourceTrain = new DataSource("docs/modelTrain.arff");
            Instances dataTrain = sourceTrain.getDataSet();
            dataTrain.setClassIndex((dataTrain.numAttributes()-1));

            FilteredClassifier fc = getFilteredClassifier();
            fc.buildClassifier(dataTrain);
            return fc;
	}
        
	public static FilteredClassifier train(Instances train, int classIndex) throws Exception
	{
            FilteredClassifier fc = getFilteredClassifier();
            train.setClassIndex(classIndex);
            fc.buildClassifier(train);
            return fc;
		
	}
        
        public static double trainSelfEval(ArrayList<Integer> array) throws Exception
	{
            arffFunctions.generateArff(array, "docs/samsung_header.txt", "model.arff");
            DataSource source = new DataSource("docs/model.arff");
            Instances data = source.getDataSet();
            data.setClassIndex((data.numAttributes()-1));

            FilteredClassifier cls = getFilteredClassifier();
            cls.buildClassifier(data);
            // evaluation
            Evaluation eval = new Evaluation(data);
            eval.evaluateModel(cls, data);
            return eval.pctCorrect();
	}
        
        public static double selfCVEval(ArrayList<Integer> array) throws Exception
	{
            arffFunctions.generateArff(array, "docs/samsung_header.txt", "model.arff");
            DataSource source = new DataSource("docs/model.arff");
            Instances data = source.getDataSet();
            data.setClassIndex((data.numAttributes()-1));

            FilteredClassifier cls = getFilteredClassifier();
            //cls.buildClassifier(data);
            // Cross Validation Evaluation
            Random random = new Random();
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(cls, data, 10, random);
            return eval.pctCorrect();
	}
	
	public static double eval(FilteredClassifier fc, Instances train, Instances test)  throws Exception
	{
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(fc, test);
		return eval.pctCorrect();
	}

	public static double evalCrossValidation(FilteredClassifier fc, Instances data) throws Exception
	{
		Random random = new Random();
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(fc, data, 10, random);
		return eval.pctCorrect();
	}
	
        public static double trainAndEval(ArrayList<Integer> trainArray, ArrayList<Integer> testArray) throws IOException, Exception{
            arffFunctions.generateArff(trainArray, "docs/samsung_header.txt", "modelTrain.arff");
            arffFunctions.generateArff(testArray, "docs/samsung_header.txt", "modelTest.arff");

            DataSource sourceTrain = new DataSource("docs/modelTrain.arff");
            DataSource sourceTest = new DataSource("docs/modelTest.arff");

            Instances dataTrain = sourceTrain.getDataSet();
            Instances dataTest = sourceTest.getDataSet();

            int classIndex = dataTrain.numAttributes()-1;
            dataTrain.setClassIndex(classIndex);
            dataTest.setClassIndex(classIndex);
		
            FilteredClassifier fc = getFilteredClassifier();
            // train
            fc.buildClassifier(dataTrain);
            // evaluation
            Evaluation eval = new Evaluation(dataTrain);
            eval.evaluateModel(fc, dataTest);
            return eval.pctCorrect();	
	}
        
	public static double trainAndEval(Instances train, Instances test, int classIndex) throws Exception{
            train.setClassIndex(classIndex);		
            FilteredClassifier fc = getFilteredClassifier();
            // train
            fc.buildClassifier(train);
            // evaluation
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(fc, test);
            return eval.pctCorrect();	
	}
        
        public static double trainAndEvalNoPrune(Instances train, Instances test, int classIndex) throws Exception{
            train.setClassIndex(classIndex);		
            FilteredClassifier fc = getFilteredClassifier();
            // train
            fc.buildClassifier(train);
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(fc, test);
            return eval.pctCorrect();	
	}
        
        public static Instances trim(Instances data, int classIndex){
                //int count = 0;
                for (int i = data.numInstances()-1; i>=0; i--){
                        //System.out.println(data.instance(i).stringValue(classIndex-1));
                        if (!data.instance(i).stringValue(classIndex-1).equals("always")){
                                //count++;
                                data.delete(i);
                        }
                }
                //System.out.println("Not ALWALS INSTANCES #: "+count);
		return data;
    }
}
