from collections import defaultdict
import getopt
import sys

tags = ["noun", "verb", "inf", "prep"]
start_tag = "phi"
end_tag = "fin"
def_prob = 0.0001
probabilities = defaultdict(lambda: defaultdict(lambda: def_prob))

def parse_probabilities(prob_file):
    with open(prob_file) as f:
        for line in f:
            parts = line.split()
            probabilities[parts[0]][parts[1]] = float(parts[2])

def gen_tags(words):
    feasible_tags = {}
    feasible_tags[-1] = [start_tag]
    for i in range(len(words)):
        feasible_tags[i] = tags
    feasible_tags[len(words)] = [end_tag]
    return feasible_tags

def forward(line):
    fwd_dp = defaultdict(lambda: defaultdict(lambda: 0))
    fwd_dp[-1][start_tag] = 1.0
    words = line.split()
    feasible_tags = gen_tags(words)
    for i in range(len(words)):
        for curr_tag in feasible_tags[i]:
            for prev_tag in feasible_tags[i - 1]:
                fwd_dp[i][curr_tag] += fwd_dp[i - 1][prev_tag] * (probabilities[curr_tag][prev_tag]) * (probabilities[words[i]][curr_tag])

    for tag in tags:
        fwd_dp[len(words)][end_tag] += fwd_dp[len(words) - 1][tag] * probabilities[end_tag][tag]
    print
    print "FORWARD ALGORITHM RESULTS"
    for i in range(len(words)):
        for tag in feasible_tags[i]:
            print "P(%s=%s) = %.10f" % (words[i], tag, fwd_dp[i][tag])

# Bigram Viterbi
def viterbi(line):
    dp = defaultdict(lambda: defaultdict(lambda: 0))
    dp[-1][start_tag] = 1.0
    bp = defaultdict(lambda: defaultdict(lambda: 0))
    words = line.split()
    feasible_tags = gen_tags(words)
    for i in range(len(words)):
        for curr_tag in feasible_tags[i]:
            for prev_tag in feasible_tags[i - 1]:
                temp_prob = dp[i - 1][prev_tag] * (probabilities[curr_tag][prev_tag]) * (probabilities[words[i]][curr_tag])
                if temp_prob > dp[i][curr_tag]:
                    dp[i][curr_tag] = temp_prob
                    bp[i][curr_tag] = prev_tag

    max_found = 0.0
    for tag in tags:
        temp_prob = dp[len(words) - 1][tag] * probabilities[end_tag][tag]
        if temp_prob > max_found:
            max_found = temp_prob
            dp[len(words)][end_tag] = max_found
            bp[len(words)][end_tag] = tag
    print
    print "PROCESSING SENTENCE: ", line
    print "FINAL VITERBI NETWORK"
    for i in range(len(words)):
        for tag in feasible_tags[i]:
            print "P(%s=%s) = %.10f" % (words[i], tag, dp[i][tag])
    print
    print "FINAL BACKPTR NETWORK"
    for i in range(len(words) - 1, 0, -1):
        for tag in feasible_tags[i]:
            print "Backptr(%s=%s) = %s" % (words[i], tag, bp[i][tag])
    print
    print "BEST TAG SEQUENCE HAS PROBABILITY = %.10f" %(dp[len(words)][end_tag])
    temp_tag = end_tag
    for i in range(len(words), 0, -1):
        print "%s -> %s" %(words[i - 1], bp[i][temp_tag])
        temp_tag = bp[i][temp_tag]

def main():
    (options, args) = getopt.getopt(sys.argv[1:], '')
    parse_probabilities(args[0])
    with open(args[1]) as test_file:
        for line in test_file:
            viterbi(line)
            forward(line)
if __name__ == "__main__":
    main()
