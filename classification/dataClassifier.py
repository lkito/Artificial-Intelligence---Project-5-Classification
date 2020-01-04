# dataClassifier.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# This file contains feature extraction methods and harness
# code for data classification

import mostFrequent
import naiveBayes
import perceptron
import perceptron_pacman
import mira
import samples
import sys
import util
from pacman import Directions, GameState

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

# Counts total amount of highlighted pixels in the top half part of the grid.
def top(datum):
    top = 0
    for y in range(DIGIT_DATUM_HEIGHT/2):
        for x in range(DIGIT_DATUM_WIDTH):
            if datum.getPixel(x, y) > 0:
                top +=1

    return top*1.0

# Counts the amount of changes of pixels on horizontal line of the grid.
# In this case, "change" means that we have a highlighted pixel after
# a non-highlighted one or vice-versa. 
def surfaceArea(datum):
    surface = 0
    for y in range(1, DIGIT_DATUM_HEIGHT):
        for x in range(1, DIGIT_DATUM_WIDTH):
            if datum.getPixel(x, y) != datum.getPixel(x - 1, y):
                surface += 1
                
    return surface

# Counts total amount of highlighted pixels in the grid.
def highlightedPixels(datum):
    pix = 0
    for y in range(DIGIT_DATUM_HEIGHT):
        for x in range(DIGIT_DATUM_WIDTH):
            if datum.getPixel(x, y) > 0:
                pix += 1
                
    return pix*1.0

# Counts the amount of "continious spaces" in the picture.
# "Continious space" is a space in which we can move freely from one point
# to another, they are made out of the non-highlighted pixels and sometimes have
# highlighted pixels around them.
def getNumContiniousSpaces(datum):
    # Set used to check if we've been on a position before
    been = set()
    spaceCount = 0
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) == 0 and (x, y) not in been:
                spaceCount += 1
                explore(been, (x, y), datum)
    return spaceCount

# From one non-highlighted pixel, explores every single neighbouring
# non-highlighted pixels recursively. Basically, explores the whole
# "continious space.
def explore(been, positionTuple, datum):
    x, y = positionTuple[0], positionTuple[1]
    
    # Memorize that this position is explored
    been.add(positionTuple)
    # If we can move to the neighbouring pixel, it's non-highlighted and we haven't
    # visited it, then visit it. Check neighbours left, right, down and up.
    if isLegalPosition(x-1, y) and datum.getPixel(x-1, y) == 0 and (x-1, y) not in been:
        explore(been, (x-1, y), datum)
    if isLegalPosition(x+1, y) and datum.getPixel(x+1, y) == 0 and (x+1, y) not in been:
        explore(been, (x+1, y), datum)
    if isLegalPosition(x, y-1) and datum.getPixel(x, y-1) == 0 and (x, y-1) not in been:
        explore(been, (x, y-1), datum)
    if isLegalPosition(x, y+1) and datum.getPixel(x, y+1) == 0 and (x, y+1) not in been:
        explore(been, (x, y+1), datum)

# Check if the coordinates are in the constraints of the grid.
def isLegalPosition(x, y):
    return x >= 0 and x < DIGIT_DATUM_WIDTH and y >= 0 and y < DIGIT_DATUM_HEIGHT



def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for this datum (datum is of type samples.Datum).

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...
        Feature n1: "numPix"
            The amount of highlighted pixels.
        Feature n2: "top"
            The ratio of highlighted pixels in the top part of the picture
            vs the total amount of highlighted pixels.
        Feature n3: "surfaceArea"
            Counts the amount of changes of pixels on horizontal lines.
            In this case, "change" means that we have a highlighted pixel after
            a non-highlighted one or vice-versa. In a way, this is a "surface area" of the number.
            E.G.: (###...##..#...) has 5 changes.
        Feature n4: "ContiniousSpaces"
            Counts the amount of "continious spaces" in the picture.
            "Continious space" is a space in which we can move freely from one point
            to another, they are made out of the non-highlighted pixels and sometimes have
            highlighted pixels around them.

            For example, normally, digits (1, 2, 3, 5, 7) have 1 continious space,
            digits (4, 6, 9, 0) have 2 continious spaces
            and digit (8) has 3 continious spaces.
        
    ##
    """
    features =  basicFeatureExtractorDigit(datum)

    # The values of these features were observed and then distributed to
    # multiple binary features. For example, if we have a feature that
    # can have a value of 55, 105 and 155, we can spread it into 3 binary
    # features: 1 - is it more than 50?, 2 - is it more than 100?,
    # 3 - is it more than 150?
    #
    # For every single one of these "main" features(which we then distribute to simple ones),
    # we only have one simple(binary) feature set to value of 1:
    # In the upper example, if we had main feature value 130, we would have
    # binary values 0, 1, 0. To make sure that binary features don't get
    # unnecessarily set to 1, I use the boolean variable.
    # (If we know that value is more than 100, we don't need to know that it's more than 50)
    flag = False
    numPix = highlightedPixels(datum)
    for i in range(20, 5, -3):
        if flag:
            features["pix" + str(i)] = 0
        else:
            features["pix" + str(i)] = numPix > i * 10
        if numPix > i * 10: flag = True
        
    # See explanation at the top of the function
    flag = False
    TVB = top(datum)/numPix
    for i in range(20, 0, -3):
        if flag:
            features["tvb" + str(i)] = 0
        else:
            features["tvb" + str(i)] = TVB > i*1.0/20
        if TVB > i*1.0/20: flag = True

    # See explanation at the top of the function
    flag = False
    surf = surfaceArea(datum)
    for i in range(200, 50, -20):
        if flag:
            features["surf" + str(i)] = 0
        else:
            features["surf" + str(i)] = surf > i
        if surf > i: flag = True

    # See explanation at the top of the function
    numSpaces = getNumContiniousSpaces(datum)
    for i in range(1, 6):
        if i == numSpaces:
            features["numSpaces" + str(i)] = 1
        else:
            features["numSpaces" + str(i)] = 0
    return features



def basicFeatureExtractorPacman(state):
    """
    A basic feature extraction function.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """
    features = util.Counter()
    for action in state.getLegalActions():
        successor = state.generateSuccessor(0, action)
        foodCount = successor.getFood().count()
        featureCounter = util.Counter()
        featureCounter['foodCount'] = foodCount
        features[action] = featureCounter
    return features, state.getLegalActions()

def enhancedFeatureExtractorPacman(state):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """

    features = basicFeatureExtractorPacman(state)[0]
    for action in state.getLegalActions():
        features[action] = util.Counter(features[action], **enhancedPacmanFeatures(state, action))
    return features, state.getLegalActions()

def enhancedPacmanFeatures(state, action):
    """
    For each state, this function is called with each legal action.
    It should return a counter with { <feature name> : <feature value>, ... }
    """
    features = util.Counter()
    
    # Feature: Is the action stop(for stop agent)
    features["stop"] = action == Directions.STOP
    # Feature: Is the game won
    features["win"] = state.isWin()
    # Feature: Is the game lost
    features["loss"] = state.isLose()
    # Feature: Current game score
    features["score"] = state.getScore()

    successor = state.generateSuccessor(0, action)
    pacmanPos = successor.getPacmanPosition()
    ghosts = successor.getGhostPositions()

    # Find the distance to the closest ghost
    closestGhostDist = 100000
    for ghost in ghosts:
        curDist = util.manhattanDistance(pacmanPos, ghost)
        closestGhostDist = min(curDist, closestGhostDist)

    # Feature: The distance to the closest ghost. When closer means better(for suicide agent)
    features["closestGhostDistPositive"] = 10.0/max(closestGhostDist, 1)
    # Feature: The distance to the closest ghost. When closer means worse(for contest agent)
    features["closestGhostDistNegative"] = closestGhostDist

    # Find the distance to the closest capsule
    capsules = successor.getCapsules()
    closestCapDist = 10000
    for cap in capsules:
        curDist = util.manhattanDistance(pacmanPos, cap)
        closestCapDist = min(curDist, closestCapDist)

    # Feature: The distance to the closest capsule. When closer means better
    features["closestCapDist"] = 10.0/max(closestCapDist, 1)
    
    # Generate list of food positions from True/False 2D array.
    foodState = state.getFood()
    food = []
    for i in range(foodState.width):
        for j in range(foodState.height):
            if foodState[i][j]: food.append((i, j))
            
    # Find the distance to the closest food
    closestFoodDist = 10000
    for f in food:
        curDist = util.manhattanDistance(pacmanPos, f)
        closestFoodDist = min(curDist, closestFoodDist)

    # Feature: The distance to the closest food. When closer means better
    features["closestFoodDist"] = 10.0/max(closestFoodDist, 1)

    return features


def contestFeatureExtractorDigit(datum):
    """
    Specify features to use for the minicontest
    """
    features =  basicFeatureExtractorDigit(datum)
    return features

def enhancedFeatureExtractorFace(datum):
    """
    Your feature extraction playground for faces.
    It is your choice to modify this.
    """
    features =  basicFeatureExtractorFace(datum)
    return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the printImage(<list of pixels>) function to visualize features.

    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features
    (see its use in the odds ratio part in runClassifier method)

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(guesses)):
    #     prediction = guesses[i]
    #     truth = testLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print rawTestData[i]
    #         break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None,self.width,self.height)
        for pix in pixels:
            try:
            # This is so that new features that you could define which
            # which are not of the form of (x,y) will not break
            # this image printer...
                x,y = pix
                image.pixels[x][y] = 2
            except:
                print "new features:", pix
                continue
        print image

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest'], default='mostFrequent')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces', 'pacman'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-g', '--agentToClone', help=default("Pacman agent to copy"), default=None, type="str")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print "Doing classification"
    print "--------------------"
    print "data:\t\t" + options.data
    print "classifier:\t\t" + options.classifier
    if not options.classifier == 'minicontest':
        print "using enhanced features?:\t" + str(options.features)
    else:
        print "using minicontest feature extractor"
    print "training set size:\t" + str(options.training)
    if(options.data=="digits"):
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorDigit
        else:
            featureFunction = basicFeatureExtractorDigit
        if (options.classifier == 'minicontest'):
            featureFunction = contestFeatureExtractorDigit
    elif(options.data=="faces"):
        printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorFace
        else:
            featureFunction = basicFeatureExtractorFace
    elif(options.data=="pacman"):
        printImage = None
        if (options.features):
            featureFunction = enhancedFeatureExtractorPacman
        else:
            featureFunction = basicFeatureExtractorPacman
    else:
        print "Unknown dataset", options.data
        print USAGE_STRING
        sys.exit(2)

    if(options.data=="digits"):
        legalLabels = range(10)
    else:
        legalLabels = ['Stop', 'West', 'East', 'North', 'South']

    if options.training <= 0:
        print "Training set size should be a positive integer (you provided: %d)" % options.training
        print USAGE_STRING
        sys.exit(2)

    if options.smoothing <= 0:
        print "Please provide a positive number for smoothing (you provided: %f)" % options.smoothing
        print USAGE_STRING
        sys.exit(2)

    if options.odds:
        if options.label1 not in legalLabels or options.label2 not in legalLabels:
            print "Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2)
            print USAGE_STRING
            sys.exit(2)

    if(options.classifier == "mostFrequent"):
        classifier = mostFrequent.MostFrequentClassifier(legalLabels)
    elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        classifier.setSmoothing(options.smoothing)
        if (options.autotune):
            print "using automatic tuning for naivebayes"
            classifier.automaticTuning = True
        else:
            print "using smoothing parameter k=%f for naivebayes" %  options.smoothing
    elif(options.classifier == "perceptron"):
        if options.data != 'pacman':
            classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
        else:
            classifier = perceptron_pacman.PerceptronClassifierPacman(legalLabels,options.iterations)
    elif(options.classifier == "mira"):
        if options.data != 'pacman':
            classifier = mira.MiraClassifier(legalLabels, options.iterations)
        if (options.autotune):
            print "using automatic tuning for MIRA"
            classifier.automaticTuning = True
        else:
            print "using default C=0.001 for MIRA"
    elif(options.classifier == 'minicontest'):
        import minicontest
        classifier = minicontest.contestClassifier(legalLabels)
    else:
        print "Unknown classifier:", options.classifier
        print USAGE_STRING

        sys.exit(2)

    args['agentToClone'] = options.agentToClone

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options

# Dictionary containing full path to .pkl file that contains the agent's training, validation, and testing data.
MAP_AGENT_TO_PATH_OF_SAVED_GAMES = {
    'FoodAgent': ('pacmandata/food_training.pkl','pacmandata/food_validation.pkl','pacmandata/food_test.pkl' ),
    'StopAgent': ('pacmandata/stop_training.pkl','pacmandata/stop_validation.pkl','pacmandata/stop_test.pkl' ),
    'SuicideAgent': ('pacmandata/suicide_training.pkl','pacmandata/suicide_validation.pkl','pacmandata/suicide_test.pkl' ),
    'GoodReflexAgent': ('pacmandata/good_reflex_training.pkl','pacmandata/good_reflex_validation.pkl','pacmandata/good_reflex_test.pkl' ),
    'ContestAgent': ('pacmandata/contest_training.pkl','pacmandata/contest_validation.pkl', 'pacmandata/contest_test.pkl' )
}
# Main harness code



def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']
    
    # Load data
    numTraining = options.training
    numTest = options.test

    if(options.data=="pacman"):
        agentToClone = args.get('agentToClone', None)
        trainingData, validationData, testData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES.get(agentToClone, (None, None, None))
        trainingData = trainingData or args.get('trainingData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][0]
        validationData = validationData or args.get('validationData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][1]
        testData = testData or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][2]
        rawTrainingData, trainingLabels = samples.loadPacmanData(trainingData, numTraining)
        rawValidationData, validationLabels = samples.loadPacmanData(validationData, numTest)
        rawTestData, testLabels = samples.loadPacmanData(testData, numTest)
    else:
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


    # Extract features
    print "Extracting features..."
    trainingData = map(featureFunction, rawTrainingData)
    validationData = map(featureFunction, rawValidationData)
    testData = map(featureFunction, rawTestData)

    # Conduct training and testing
    print "Training..."
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print "Validating..."
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
    print "Testing..."
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

    # do odds ratio computation if specified at command line
    if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
        label1, label2 = options.label1, options.label2
        features_odds = classifier.findHighOddsFeatures(label1,label2)
        if(options.classifier == "naiveBayes" or options.classifier == "nb"):
            string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
        else:
            string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

        print string3
        printImage(features_odds)

    if((options.weights) & (options.classifier == "perceptron")):
        for l in classifier.legalLabels:
            features_weights = classifier.findHighWeightFeatures(l)
            print ("=== Features with high weight for label %d ==="%l)
            printImage(features_weights)

if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runClassifier(args, options)
