import java.util.ArrayList;
import java.util.Collections;


public class randomShuffle2 {
	
    public static void randomShuffle(int TIMES) throws Exception {
		
            double accuracy1Max = 0;
            double accuracy2Max = 0;
            ArrayList<Integer> userArray = user.getUserIDArray();
            int userArraySize = userArray.size();
		 
            for (int expTimes = 0; expTimes < TIMES; expTimes++){
                System.out.println( "Round " + (expTimes+1) );
                // shuffle the list
                Collections.shuffle(userArray);
                //System.out.println("ArrayU: " + userArray);
                ArrayList<Integer> array1 = new ArrayList<>(userArray.subList(0, Math.round(userArraySize/2)));
                ArrayList<Integer> array2 = new ArrayList<>(userArray.subList(Math.round(userArraySize/2), userArraySize));
                Collections.sort(array1);
                Collections.sort(array2);

                System.out.println("Array1: " + array1);
                System.out.println("Array2: " + array2);

                double accuracy1 = wekaFunctions.trainAndEval(array1, userArray);
                double accuracy2 = wekaFunctions.trainAndEval(array2, userArray);
                System.out.println("Array1's accuracy: " + accuracy1);
                System.out.println("Array2's accuracy: " + accuracy2);
                //System.out.println((accuracy1 > accuracy1Max));
                //System.out.println((accuracy2 > accuracy2Max));
                if (  (accuracy1 > accuracy1Max) == true && (accuracy2 > accuracy2Max) == true ){
                        accuracy1Max = accuracy1;
                        accuracy2Max = accuracy2;
                }

                System.out.println("Array1's Max accuracy: " + accuracy1Max);
                System.out.println("Array2's Max accuracy: " + accuracy2Max);

            }
            System.out.println("Final Array1's Max accuracy: " + accuracy1Max);
            System.out.println("Final Array2's Max accuracy: " + accuracy2Max);

	}

    public static void main(String[] args) throws Exception {
	randomShuffle(10);
    }   
}
