# Sudi

## sudi

* `sudi` is a machine learning algorithm that can idenify the part of speach of inputted sentences
* `sudi` can be trained using a .txt file of sample sentences. For this project I used sentences that were drawn from the Brown University and Dartmouth College corpus'. Having been trained, `sudi` can then be fed sentences that it has not seen before and will determine the part of speech of each word in the sentence with up to 91% accuracy. 

### Psudeocode

`sudi` implements a three methods: 1. training 2. tagging 3. decode which are all contained in the same .java file

The `training` method follows the following psudeocode: 
```
initialize reference to BufferedReader object for sentences .txt file
initialize reference to BufferedReader object for part of speach .txt file
add start verted to global Graph transitions
loop through each line in sentences .txt file
	parse line into array
	parse part of speech tags into array
	loop through each tag in array
		insert tag into transitions graph
		insert proper directed edge if necessary
		insert word into global HashMap observations if necessary
	iterate through each vertex in transitions
		correct probabilies of each transition (edge label)
	iterate through each key in observations
		correct probabilies of each observation

```

The `tagging` method follows uses the implements the Vierbi decoding algorithm to estimate the part of speech of words in unknown sentences. It follows the following psudeocode: 
```
intialze reference to BufferedReader object 
intiialize ArrayList finalPath
while there are more lines to read from testInputSentence:
    	read a line from testInputSentence and store it in testSentence
    	split the testTag by spaces and store the resulting array in tags
    	split the testSentence by spaces and store the resulting array in words
    	create an empty set called currStates
    	create an empty map called currScores
    	create an empty map called pathTransitions

    	add the 'start' state to the currStates set
    	set the score of the 'start' state to 0.0 in the currScores map

    	for each word in words:
        	initialize empty Set called nextStates
        	initialize empty Map called nextScores
        	intiialize empty Set called pathTransitions[word]

        	for each currState in currStates
            		for each nextState in the outgoing neighbors of currState based on the 'transitions' data:
                		add nextState to the nextStates set
                		calculate the score for transitioning from currState to nextState:
                    		retrieve the label associated with the transition from currState to nextState
                    		add the label score to the current score of currState

                    		if the observations for nextState contains an entry for the current word:
                        		add the corresponding observation score to the total score
                    			otherwise, add a default score for unknown observations

               			if nextState is not present in the nextScores
                   			add nextState to the nextScores map with the calculated score as its value
                    			add the transition (nextState + " " + currState) to the pathTransitions[word] set

                		if the calculated score is greater than the score currently stored for nextState in the nextScores
                    			update the score of nextState in the nextScores map with the calculated score
                    			for each transition in the pathTransitions[word]
                       			split the transition by spaces and store the resulting array in transArray
                       			if the first element of transArray matches nextState
                           		remove the transition from the pathTransitions[word]
                            			add the updated transition (nextState + " " + currState) to the pathTransitions[word] set
                            			exit the loop

        		set currStates to the nextStates set
        		set currScores to the nextScores map
```

The `decode` method backtracks from currScores to get the most likely path (ie. implemented the Hidden Markov Model approach) to determine the most likely part of speech for each work in the unknown sentence. 

```
initialize empty ArrayList path
set the initial highScore to -1 times the maximum value of Double
initialize the highPOS variable as an empty string

for each POS in the keys of the currScores map:
   	if the score associated with the current POS is greater than highScore:
        	update highScore with the score of the current POS
        	update highPOS with the current POS

	set the current POS as highPOS, starting with the POS with the highest probability at the end of Viterbi decoding
	add the current POS to the beginning of the path ArrayList
	set the index to the length of the sentence minus 1
	while the current POS is not equal to "#" and the index is greater than 0:
    		set the current word as the word at the current index in the sentence
    		for each transition in the set of transitions retrieved from pathTransitions for the current word:
        		split the transition into an array called transArray, using a space as the delimiter
        		if the first element of transArray (nextScore) is equal to the current POS:
           			update the current POS to the second element of transArray (currScore)
            			add the current POS to the beginning of the path ArrayList
            			exit the loop

    			decrease the index by 1

return the path ArrayList

```


