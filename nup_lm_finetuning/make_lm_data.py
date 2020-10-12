import re
import argparse

def make_lm_data(input_file, output_file):
    """Reads a tab separated value file."""
    with open(input_file, 'r', encoding='utf-8') as rf, open(output_file, 'w', encoding='utf-8') as wf:
        # cnt = 0
        for line in rf:
            # cnt += 1
            line = line.strip().split('\t')
            if line[0] == '1':
                for i in range(1, len(line) - 1):
                    line[i] = re.compile('[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f]').sub(' ', line[i])
                    if len(line[i].strip()) >= 1:
                        wf.write(line[i] + '\n')
                line[-1] = re.compile('[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f]').sub(' ', line[-1])
                if len(line[-1].strip()) >= 1:
                    wf.write(line[-1])

            wf.write('\n')

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_file",
                        default=None,
                        type=str,
                        help="The input data dir.")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        help="The output data dir.")

    args = parser.parse_args()

    make_lm_data(args.data_file, args.output_file)

if __name__ == "__main__":
    main()