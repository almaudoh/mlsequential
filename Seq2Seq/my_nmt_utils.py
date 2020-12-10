import torch
import pandas as pd


# def _load_dataset():
#     dataset = [
#         ('9 may 1998', '1998-05-09'),
#         ('10.09.70', '1970-09-10'),
#         ('4/28/90', '1990-04-28'),
#         ('thursday january 26 1995', '1995-01-26'),
#         ('monday march 7 1983', '1983-03-07'),
#         ('sunday may 22 1988', '1988-05-22'),
#         ('tuesday july 8 2008', '2008-07-08'),
#         ('08 sep 1999', '1999-09-08'),
#         ('1 jan 1981', '1981-01-01'),
#         ('monday may 22 1995', '1995-05-22')
#     ]

def _load_dataset():
    dataset = pd.read_excel('date data.xlsx')

    human_vocab = {
        ' ': 0, '.': 1, '/': 2, '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10, '8': 11,
        '9': 12, 'a': 13, 'b': 14, 'c': 15, 'd': 16, 'e': 17, 'f': 18, 'g': 19, 'h': 20, 'i': 21, 'j': 22,
        'l': 23, 'm': 24, 'n': 25, 'o': 26, 'p': 27, 'r': 28, 's': 29, 't': 30, 'u': 31, 'v': 32, 'w': 33,
        'y': 34, '<unk>': 35, '<pad>': 36
    }
    machine_vocab = {'-': 0, '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10}
    inv_machine_vocab = {0: '-', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9'}

    # human_vocab = {chr(ind): ind - 31 for ind in range(31, 127)}
    # machine_vocab = {chr(ind+48): ind for ind in range(10)}
    # machine_vocab['-'] = 10
    # inv_machine_vocab = {ind: char for char, ind in machine_vocab.items()}

    return dataset, human_vocab, machine_vocab, inv_machine_vocab


def encode_strings(strings, vocab, size, use_onehot=True):
    letters = []
    for string in strings:
        item = [vocab[c] if c in vocab else vocab['<unk>'] for c in string]
        if len(item) < size:
            item += [vocab['<pad>']] * (size - len(item))
        letters.append(item)

    encoded = torch.tensor(letters)

    if use_onehot:
        onehot = torch.FloatTensor(len(strings), size, len(vocab))
        onehot.zero_()
        return onehot.scatter_(2, encoded.view(len(strings), -1, 1), 1)

    else:
        return encoded


def decode_strings(output, inv_machine_vocab):
    string = ''
    for val in output:
        string += inv_machine_vocab[val.item()]

    return string


