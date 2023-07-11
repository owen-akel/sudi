import java.util.*;
import java.io.*;
/**
 * Sudi — Machine learning algorithm determining the POS of inputted sentences
 *
 * @author Owen Akel
 */
public class Sudi {
    private Double unknown = -100.0;
    private Graph<String, Double> transitions;
    private Map<String, Map<String, Double>> observations;
    String start;

    /**
     * Constructor
     */
    public Sudi(){
        this.start = "#";
        transitions = new AdjacencyMapGraph<>();
        observations = new HashMap<>();
    }

    /**
     * Train Sudi using sentences and their corresponding tags
     * @param sentencesPathName
     * @param tagsPathName
     * @throws IOException
     */
    public void training(String sentencesPathName, String tagsPathName) throws IOException{
        BufferedReader sentencesInput = new BufferedReader(new FileReader(sentencesPathName));
        BufferedReader tagsInput = new BufferedReader(new FileReader(tagsPathName));
        String sentenceLine = "";
        // Insert start
        transitions.insertVertex(start);
        // Loading sentencesFile and tagsFile into graph
        while((sentenceLine = sentencesInput.readLine()) != null){
            String[] words = sentenceLine.split(" ");
            String[] tags = tagsInput.readLine().split(" ");
            for(int i = 0; i<tags.length; i++){
                String tag = tags[i];
                transitions.insertVertex(tag);
                if(i>0){
                    String prevTag = tags[i-1];
                    if(transitions.hasEdge(prevTag, tag)){
                        Double tagCount = transitions.getLabel(prevTag, tag);
                        transitions.insertDirected(prevTag, tag, tagCount+1);
                    }
                    else transitions.insertDirected(prevTag, tag, 1.0);
                }
                else{
                    if(transitions.hasEdge(start, tag)){
                        Double tagCount = transitions.getLabel(start, tag);
                        transitions.insertDirected(start, tag, tagCount+1);
               
                    else transitions.insertDirected(start, tag, 1.0);
                } 
                String word = words[i].toLowerCase();
                if(observations.containsKey(tag)){
                    if(observations.get(tag).containsKey(word)){
                        Double wordCount = observations.get(tag).get(word);
                        observations.get(tag).put(word, wordCount+1);
                    }
                    else observations.get(tag).put(word, 1.0);
                }
                else{
                    observations.put(tag, new HashMap<String, Double>());
                    observations.get(tag).put(word, 1.0);
                }
            }
        }

        //Loop through transitions and correct edge labels
        for(String vertex: transitions.vertices()){
            Double denominator = 0.0;
            for(String neighbor: transitions.outNeighbors(vertex)){
                denominator += transitions.getLabel(vertex, neighbor);
            }
            for(String neighbor: transitions.outNeighbors(vertex)) {
                Double numerator = transitions.getLabel(vertex, neighbor);
                transitions.insertDirected(vertex, neighbor, Math.log(numerator/denominator));
            }
        }

        //Loop through observations and correct probability
        for(String POS: observations.keySet()){
            Double denominator = 0.0;
            for(String word: observations.get(POS).keySet()){
                denominator += observations.get(POS).get(word);
            }
            for(String word: observations.get(POS).keySet()){
                Double numerator = observations.get(POS).get(word);
                observations.get(POS).put(word, Math.log(numerator/denominator));
            }
        }
    }

    /**
     * Implements Viterbi decoding algorithm to estimate what POS words in unknown
     * sentences are based on transitions and observations
     *
     * @param testSentencesPathName
     * @return
     * @throws IOException
     */
    public ArrayList<ArrayList<String>> tagging(String testSentencesPathName) throws IOException{
        BufferedReader testInputSentence = new BufferedReader(new FileReader(testSentencesPathName));
        String testSentence = "";
        String testTag = "";
        ArrayList<ArrayList<String>> finalPath = new ArrayList<>();
        while((testSentence = testInputSentence.readLine()) != null){
            String[] tags = testTag.split(" ");
            String[] words = testSentence.split(" ");
            Set<String> currStates = new HashSet<>();
            Map<String, Double> currScores = new HashMap<>();
            Map<String,Set<String>> pathTransitions = new HashMap<>();
            currStates.add(start);
            currScores.put(start, 0.0);
            for(String word: words){
                Set<String> nextStates = new HashSet<>();
                Map<String, Double> nextScores = new HashMap<>();
                pathTransitions.put(word, new HashSet<>());
                for(String currState: currStates){
                    for(String nextState: transitions.outNeighbors(currState)){
                        nextStates.add(nextState);
                        Double nextScore = currScores.get(currState) + transitions.getLabel(currState, nextState);
                        if(observations.get(nextState).containsKey(word)){
                            nextScore += observations.get(nextState).get(word);
                        }
                        else{
                            nextScore += unknown;
                        }
                        if(!(nextScores.containsKey(nextState))){
                            nextScores.put(nextState, nextScore);
                            pathTransitions.get(word).add(nextState+" "+currState);
                        }
                        if(nextScore>nextScores.get(nextState)){
                            nextScores.put(nextState, nextScore);
                            for(String trans: pathTransitions.get(word)){
                                String[] transArray = trans.split(" ");
                                if(transArray[0].equals(nextState)){
                                    pathTransitions.get(word).remove(trans);
                                    pathTransitions.get(word).add(nextState+" "+currState);
                                    break;
                                }
                            }
                        }
                    }
                }
                currStates = nextStates;
                currScores = nextScores;
            }
            ArrayList<String> path = decode(currScores, pathTransitions, words);
            finalPath.add(path);
        }
        return finalPath;
    }

    /**
     * Backtrack from currScores to find path
     * @param currScores
     * @param pathTransitions
     * @param sentence
     * @return
     */
    public ArrayList<String> decode(Map<String, Double> currScores, Map<String, Set<String>> pathTransitions, String[] sentence){
        ArrayList<String> path = new ArrayList<String>();
        Double highScore = -1*Double.MAX_VALUE;
        String highPOS = "";
        for(String POS: currScores.keySet()){
            if(currScores.get(POS)>highScore){
                highScore = currScores.get(POS);
                highPOS = POS;
            }
        }
        // Backtrack through pathTransitions to get path
        String currPOS = highPOS; // Start with POS w/ highest probability @ end of viterbi decoding
        path.add(currPOS);
        int index = sentence.length-1;
        while(!(currPOS.equals("#")) && index>0){
            String word = sentence[index];
            for(String trans: pathTransitions.get(word)){
                String[] transArray = trans.split(" "); // Split trans such that index 0: nextScore, index 1: currScore
                if(transArray[0].equals(currPOS)){
                    currPOS = transArray[1]; // Set currPOS equal to currScore of highest probability nextScore
                    path.add(0, currPOS); // Add corrPOS to path
                    break;
                }
            }
            index--;
        }
        return path;
    }

    /**
     * Calculates % correct: tags from tagging vs. actual
     *
     * @param testTagsPathName
     * @param testAns
     * @return
     * @throws IOException
     */
    public static Double testing(String testTagsPathName, ArrayList<ArrayList<String>> testAns) throws IOException{
        BufferedReader input = new BufferedReader(new FileReader(testTagsPathName));
        String tagLine = "";
        Double totalCorrect = 0.0;
        Double totalTags = 0.0;
        int line = 0;
        while((tagLine = input.readLine()) != null){
            String[] actTags = tagLine.split(" ");
            ArrayList<String> predTags = testAns.get(line);
            for(int i = 0; i<actTags.length; i++){
                totalTags += 1;
                if(actTags[i].equals(predTags.get(i))) totalCorrect += 1;
            }
            line += 1;
        }
        return 100*(totalCorrect/totalTags);
    }

    public static void main(String[] args) {
        // Sudi testHardcode = new Sudi(); — used for testing purposes to check accuracy of training
        Sudi testExample = new Sudi();
        Sudi testSimple = new Sudi();
        Sudi testBrown = new Sudi();
        try {
            // Hardcoded graph — used for testing purposes to check accuracy of training
            /*
            testHardcode.transitions.insertVertex("#");
            testHardcode.transitions.insertVertex("MOD");
            testHardcode.transitions.insertVertex("PRO");
            testHardcode.transitions.insertVertex("VD");
            testHardcode.transitions.insertVertex("N");
            testHardcode.transitions.insertVertex("NP");
            testHardcode.transitions.insertVertex("DET");
            testHardcode.transitions.insertVertex("V");
            testHardcode.transitions.insertVertex(".");
            testHardcode.transitions.insertDirected("#", "MOD", -2.3);
            testHardcode.transitions.insertDirected("#", "DET", -.9);
            testHardcode.transitions.insertDirected("#", "PRO", -1.2);
            testHardcode.transitions.insertDirected("#", "NP", -1.6);
            testHardcode.transitions.insertDirected("MOD", "V", -.7);
            testHardcode.transitions.insertDirected("MOD", "PRO", -.7);
            testHardcode.transitions.insertDirected("PRO", "V", -.5);
            testHardcode.transitions.insertDirected("PRO", "VD", -1.6);
            testHardcode.transitions.insertDirected("PRO", "MOD", -1.6);
            testHardcode.transitions.insertDirected("VD", "DET", -1.1);
            testHardcode.transitions.insertDirected("VD", "PRO", -.4);
            testHardcode.transitions.insertDirected("N", "VD", -1.4);
            testHardcode.transitions.insertDirected("N", "V", -1.9);
            testHardcode.transitions.insertDirected("DET", "N", 0.0);
            testHardcode.transitions.insertDirected("NP", "VD", -.7);
            testHardcode.transitions.insertDirected("NP", "V", -.7);
            testHardcode.transitions.insertDirected("V", "PRO", -1.9);
            testHardcode.transitions.insertDirected("V", "DET", -.2);
            Map<String, Double> wordsN = new HashMap<>();
            wordsN.put("color", -2.4);
            wordsN.put("cook", -2.4);
            wordsN.put("fish", -1.0);
            wordsN.put("jobs", -2.4);
            wordsN.put("mine", -2.4);
            wordsN.put("saw", -1.7);
            wordsN.put("uses", -2.4);
            testHardcode.observations.put("N", wordsN);
            Map<String, Double> wordsMOD = new HashMap<>();
            wordsMOD.put("can", -1.7);
            wordsMOD.put("will", -1.7);
            testHardcode.observations.put("MOD", wordsMOD);
            Map<String, Double> wordsV = new HashMap<>();
            wordsV.put("color", -2.1);
            wordsV.put("cook", -1.4);
            wordsV.put("eats", -2.1);
            wordsV.put("fish", -2.1);
            wordsV.put("has", -1.4);
            wordsV.put("uses", -2.1);
            testHardcode.observations.put("V", wordsV);
            Map<String, Double> wordsDET = new HashMap<>();
            wordsDET.put("a", -1.3);
            wordsDET.put("many", -1.7);
            wordsDET.put("one", -1.7);
            wordsDET.put("the", -1.0);
            testHardcode.observations.put("DET", wordsDET);
            Map<String, Double> wordsPRO = new HashMap<>();
            wordsPRO.put("I", -1.9);
            wordsPRO.put("many", -1.9);
            wordsPRO.put("me", -1.9);
            wordsPRO.put("mine", -1.9);
            wordsPRO.put("you", -.8);
            testHardcode.observations.put("PRO", wordsPRO);
            Map<String, Double> wordsVD = new HashMap<>();
            wordsVD.put("saw", -1.1);
            wordsVD.put("were", -1.1);
            wordsVD.put("wore", -1.1);
            testHardcode.observations.put("VD", wordsVD);
            Map<String, Double> wordsNP = new HashMap<>();
            wordsNP.put("jobs", -.7);
            wordsNP.put("will", -.7);
             */
            // Testing on example sentences
            testExample.training("inputs/example-sentences.txt", "inputs/example-tags.txt");
            ArrayList<ArrayList<String>> testExampleAns = testExample.tagging("inputs/example-test-sentences.txt");
            System.out.println(testExample.testing("inputs/example-test-tags.txt", testExampleAns));
            // Testing on simple files: ~87% accurate
            testSimple.training("inputs/simple-train-sentences.txt", "inputs/simple-train-tags.txt");
            ArrayList<ArrayList<String>> testSimpleAns = testSimple.tagging("inputs/simple-test-sentences.txt");
            System.out.println(testSimple.testing("inputs/simple-test-tags.txt", testSimpleAns));

            // Testing on brown files: ~91% accurate
            testBrown.training("inputs/brown-train-sentences.txt", "inputs/brown-train-tags.txt");
            ArrayList<ArrayList<String>> testBrownAns = testBrown.tagging("inputs/brown-test-sentences.txt");
            System.out.println(testBrown.testing("inputs/brown-test-tags.txt", testBrownAns));
        }
        catch(Exception e){
            System.out.println(e);
        }
    }
}
