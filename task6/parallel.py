import array
import random

batchSize = 1
inputSize = 32 * 32
hiddenSize1 = 16 * 16
hiddenSize2 = 4 * 4
outputSize = 1

# Generate float array with random numbers
w1 = array.array('f', (random.random() for _ in range(inputSize * hiddenSize1)))
w2 = array.array('f', (random.random() for _ in range(hiddenSize1 * hiddenSize2)))
w3 = array.array('f', (random.random() for _ in range(hiddenSize2 * outputSize)))

b1 = array.array('f', (random.random() for _ in range(hiddenSize1)))
b2 = array.array('f', (random.random() for _ in range(hiddenSize2)))
b3 = array.array('f', (random.random() for _ in range(outputSize)))

h_input = array.array('f', (random.random() for _ in range(inputSize)))
# Open output file
with open('wb.bin', 'wb') as output_file:
    # Write data from float array to output file
    w1.tofile(output_file)
    w2.tofile(output_file)
    w3.tofile(output_file)
    b1.tofile(output_file)
    b2.tofile(output_file)
    b3.tofile(output_file)

with open('input.bin', 'wb') as output_file:
    h_input.tofile(output_file)