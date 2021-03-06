/**
 * @author Yang
 */

import java.util.ArrayList;
import java.util.Collections;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TwoArray {
	
	private static final int ROUNDS = 1;
	
	public static ArrayList<Integer> array1 = new ArrayList<>();
	public static ArrayList<Integer> array2 = new ArrayList<>();
	
	public static int classIndex;
	public static DataSource source;
	private static DataSource source1;
	private static DataSource source2;
	
	public static Instances test;
	public static Instances data1;
	public static Instances data2;
	
	public static double maxCorrectPercentage = 0;
	public static FilteredClassifier fc1 = new FilteredClassifier();
	public static FilteredClassifier fc2 = new FilteredClassifier();

	public static FilteredClassifier maxFc1 = new FilteredClassifier();
	public static FilteredClassifier maxFc2 = new FilteredClassifier();

	public static double Array1Accuracy = 0;
	public static double Array2Accuracy = 0;

	public static int finalArray1Size = 0;
	public static int finalArray2Size = 0;

	public static void shuffle(ArrayList<Integer> userArray) throws Exception {
		
		int shuffleTimes = 0;
		int userArraySize = userArray.size();
		source = new DataSource("docs/samsung.arff");	
		test = source.getDataSet();
		classIndex = test.numAttributes()-1;
                //System.out.println("classIndex:"+ classIndex);
		test.setClassIndex(classIndex);
		
		System.out.println("2 arrays Start Shuffling...");
		shuffleTimes = 0;
                
		do{
			System.out.println("Shuffling array 1 and 2 Time: " + shuffleTimes);
			Collections.shuffle(userArray);
			array1 = new ArrayList<>(userArray.subList(0, Math.round(userArraySize/2)));
			array2 = new ArrayList<>(userArray.subList(Math.round(userArraySize/2), userArraySize));

			Collections.sort(array1);
			Collections.sort(array2);
			
			arffFunctions.generateArff(array1, "docs/samsung_header.txt", "model1.arff");
			arffFunctions.generateArff(array2, "docs/samsung_header.txt", "model2.arff");
			
			source1 = new DataSource("docs/model1.arff");
			source2 = new DataSource("docs/model2.arff");
			
			data1 = source1.getDataSet();
			data2 = source2.getDataSet();
			
			data1.setClassIndex(classIndex);
			data2.setClassIndex(classIndex);
		
			fc1 = wekaFunctions.train(data1, classIndex);
			fc2 = wekaFunctions.train(data2, classIndex);
                        
			System.out.println("fc1 size: " + fc1.numElements());
			System.out.println("fc2 size: " + fc2.numElements());

			shuffleTimes++;
		}while ( (fc1.numElements() == 1) && (fc2.numElements() == 1) );		
		System.out.println("2 arrays end Shuffling...");
        }
	

        public static void converge() throws Exception{       
                for (int expTimes = 0; expTimes < 200; expTimes++){
			
			//Array1Accuracy = wekaFunctions.evalCrossValidation(fc1, data1);
                        Array1Accuracy = wekaFunctions.eval(fc1, data1, data1);
			//Array2Accuracy = wekaFunctions.evalCrossValidation(fc2, data2);
                        Array2Accuracy = wekaFunctions.eval(fc2, data2, data2);
                        System.out.println("=============================================================================");
			System.out.println("Iteration: " + expTimes);
			//System.out.println("fc1:\n " + fc1);
			//System.out.println("fc2:\n " + fc2);
			//System.out.println("fc3:\n " + fc3);
			System.out.println("fc1 size: " + fc1.numElements() + "\t" +
					"Array1's size: " + data1.numInstances() + "\t" +
					"Array1's accuracy: " + Array1Accuracy);
			System.out.println("fc2 size: " + fc2.numElements() + "\t" +
					"Array2's size: " + data2.numInstances() + "\t" +
					"Array2's accuracy: " + Array2Accuracy);

				
                        ArrayList<Integer> array1Backup = new ArrayList<>(array1);  //BACKUPS HAVE BEEN MADE JUST IN CASE WE NEED TO PUT IT BACK IN THE SAME ARRAY
                        ArrayList<Integer> array2Backup = new ArrayList<>(array2);

                        array1.clear();
                        array2.clear();
	
                        for (int i = 0; i < test.numInstances(); i = i + 12) {
                                int userID = (int)test.instance(i).value(0);
                                //System.out.println("#"+(i+12)/12+" ID:"+userID);
                                Instances user = new Instances(test, i, 12);
                                double accuracy1 = 0;
                                double accuracy2 = 0;

                                accuracy1 = wekaFunctions.eval(fc1, data1, user);
                                accuracy2 = wekaFunctions.eval(fc2, data2, user);
                                //System.out.println(accuracy1);
                                //System.out.println(accuracy2);
                                //System.out.println();
                                if (accuracy1 > accuracy2){
                                        array1.add(userID);
                                }
                                else if (accuracy1 < accuracy2){
                                        array2.add(userID);
                                }
                                else if (accuracy1 == accuracy2) {
                                        //System.out.println(array1Backup.contains(userID));
                                        if (array1Backup.contains(userID)){
                                                array1.add(userID);
                                        }
                                        else
                                                array2.add(userID);
                                }	

                                //System.out.println("===================");
                        }
				//System.out.println("EqualAccuracyUser#: " + equaAccurayUser);
                        //System.out.println("========================================================================================================================================================");
                        //equaAccurayUser = 0;
			if ( array1.equals(array1Backup) && array2.equals(array2Backup) ){
				double accuracy = data1.numInstances()*Array1Accuracy/1359600 + data2.numInstances()*Array2Accuracy/1359600;
				System.out.println("*****************************************************************************");
				System.out.println("Arrays converged within " + expTimes + " iterations");
				System.out.println("Cumulated Correct Percentage: " +  accuracy);
				System.out.println("*****************************************************************************");
				if (accuracy > maxCorrectPercentage){
					maxCorrectPercentage = accuracy;
					maxFc1 = fc1;
					maxFc2 = fc2;
					finalArray1Size = data1.numInstances();
					finalArray2Size = data2.numInstances();
					
				}
				break;
			}
                        arffFunctions.generateArff(array1, "docs/samsung_header.txt", "model1.arff");
                        arffFunctions.generateArff(array2, "docs/samsung_header.txt", "model2.arff");

                        source1 = new DataSource("docs/model1.arff");
                        source2 = new DataSource("docs/model2.arff");

                        data1 = source1.getDataSet();
                        data2 = source2.getDataSet();

                        data1.setClassIndex(classIndex);
                        data2.setClassIndex(classIndex);

                        fc1 = wekaFunctions.train(data1, classIndex);
                        fc2 = wekaFunctions.train(data2, classIndex);
                }
        }
			
	public static void main(String[] args) throws Exception {
		for (int i = 0; i < 1000; i++){
			ArrayList<Integer> userArray = arffFunctions.getUserIDArray("docs/samsung.arff");
			shuffle(userArray);
			converge();
		}
		System.out.println("*****************************************************************************");
		System.out.println("Final Statistics:\n");
		System.out.println("maxfc1:\n"+maxFc1+"maxfc2:\n"+maxFc2);
		System.out.println("Array1's size: " +  finalArray1Size + "\t" + "accuracy: " + Array1Accuracy);
		System.out.println("Array2's size: " +  finalArray2Size + "\t" + "accuracy: " + Array2Accuracy);
		System.out.println("Max Correct Percentage: " +  maxCorrectPercentage);
		System.out.println("*****************************************************************************");
        }
}