import sys
import getopt
import os
import math
import operator


class Perceptron:
    class TrainSplit:
        """Represents a set of training/testing data. self.train is a list of Examples, as is self.test.
        """

        def __init__(self):
            self.train = []
            self.test = []

    class Example:
        """Represents a document with a label. klass is 'pos' or 'neg' by convention.
           words is a list of strings.
        """

        def __init__(self):
            self.klass = ''
            self.words = []

    def __init__(self):
        """Perceptron initialization"""
        # in case you found removing stop words helps.
        self.stopList = set(self.readFile('../data/english.stop'))
        self.numFolds = 10
        self.vocab = {}
        self.count = 0
        self.training_inputs = []
        self.training_outputs = []
        self.weights = []
        self.avg_weights = []
        self.bias = 0
        self.avg_bias = 0

    #############################################################################
    # TODO TODO TODO TODO TODO
    # Implement the Perceptron classifier with
    # the best set of features you found through your experiments with Naive Bayes.

    def classify(self, words):
        """ TODO
          'words' is a list of words to classify. Return 'pos' or 'neg' classification.
        """
        words = set(words)
        test_doc = []
        for word in words:
            if word in self.vocab:
                test_doc.append(self.vocab[word])
        guess_op = self.guess(test_doc, self.bias)
        if guess_op < 0:
            return 'neg'
        else:
            return 'pos'

    def addExample(self, klass, words):
        """
         * TODO
         * Train your model on an example document with label klass ('pos' or 'neg') and
         * words, a list of strings.
         * You should store whatever data structures you use for your classifier
         * in the Perceptron class.
         * Returns nothing
        """
        words = set(words)
        if (klass == 'pos'):
            self.training_outputs.append(1.0)
        else:
            self.training_outputs.append(-1.0)
        given_input = []
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = self.count
                given_input.append(self.count)
                self.count += 1
            else:
                given_input.append(self.vocab[word])
        self.training_inputs.append(given_input)
        # Write code here

    def guess(self, input, bias):
        guess = 0.0 + bias
        for feature_index in input:
            guess += self.weights[feature_index]
        return guess

    def update_weights(self, c, input_index):
        input = self.training_inputs[input_index]
        output = self.training_outputs[input_index]
        for feature_index in input:
            self.weights[feature_index] += output
            self.avg_weights[feature_index] += c * output
        self.bias += output
        self.avg_bias += c * output

    def initialize_parameters(self):
        vocab_size = self.count
        self.weights = [0] * vocab_size
        self.avg_weights = [0] * vocab_size
        self.bias = 0
        self.avg_bias = 0

    def train(self, split, iterations):
        """
        * TODO
        * iterates through data examples
        * TODO
        * use weight averages instead of final iteration weights
        """
        for example in split.train:
            words = example.words
            self.addExample(example.klass, words)

        self.initialize_parameters()
        N = len(split.train)
        c = 1
        num_iters = 100
        for i in range(0, num_iters):
            for n in range(0, N):
                input = self.training_inputs[n]
                guess_op = self.guess(input, self.bias)
                if guess_op * self.training_outputs[n] <= 0:
                    self.update_weights(c, n)
                c += 1
        self.weights = [(self.weights[i] - self.avg_weights[i]/c) for i in range(0, len(self.weights))]
        self.bias -= self.avg_bias/c

    # END TODO (Modify code beyond here with caution)
    #############################################################################


    def readFile(self, fileName):
        """
         * Code for reading a file.  you probably don't want to modify anything here,
         * unless you don't like the way we segment files.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        result = self.segmentWords('\n'.join(contents))
        return result

    def segmentWords(self, s):
        """
         * Splits lines on whitespace for file reading
        """
        return s.split()

    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        for fileName in posTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            example.klass = 'pos'
            split.train.append(example)
        for fileName in negTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            example.klass = 'neg'
            split.train.append(example)
        return split

    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        # for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                example.klass = 'pos'
                if fileName[2] == str(fold):
                    split.test.append(example)
                else:
                    split.train.append(example)
            for fileName in negTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                example.klass = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(example)
                else:
                    split.train.append(example)
            splits.append(split)
        return splits

    def filterStopWords(self, words):
        """Filters stop words."""
        filtered = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                filtered.append(word)
        return filtered


def test10Fold(args):
    pt = Perceptron()

    iterations = int(args[1])
    splits = pt.crossValidationSplits(args[0])
    avgAccuracy = 0.0
    fold = 0
    for split in splits:
        classifier = Perceptron()
        accuracy = 0.0
        classifier.train(split, iterations)

        for example in split.test:
            words = example.words
            guess = classifier.classify(words)
            if example.klass == guess:
                accuracy += 1.0

        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print '[INFO]\tAccuracy: %f' % avgAccuracy


def classifyDir(trainDir, testDir, iter):
    classifier = Perceptron()
    trainSplit = classifier.trainSplit(trainDir)
    iterations = int(iter)
    classifier.train(trainSplit, iterations)
    testSplit = classifier.trainSplit(testDir)
    # testFile = classifier.readFile(testFilePath)
    accuracy = 0.0
    for example in testSplit.train:
        words = example.words
        guess = classifier.classify(words)
        if example.klass == guess:
            accuracy += 1.0
    accuracy = accuracy / len(testSplit.train)
    print '[INFO]\tAccuracy: %f' % accuracy


def main():
    (options, args) = getopt.getopt(sys.argv[1:], '')

    if len(args) == 3:
        classifyDir(args[0], args[1], args[2])
    elif len(args) == 2:
        test10Fold(args)


if __name__ == "__main__":
    main()
