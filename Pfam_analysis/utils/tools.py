import torch


def create_seqs(X: torch.FloatTensor) -> list:


    # int to aa label
    int2token = {ii:label for ii, label in enumerate('ACDEFGHIKLMNPQRSTVWY-')}

    # convert one hot encoded sequences into amino acids
    num_seq = [
            list(seq) for seq in torch.argmax(X, dim = -1).cpu().numpy()[:,-1,:]
    ]

    # temp aa seq list
    temp_aa_seq = []
    # convert num to str
    for seq in num_seq:
        temp_aa_seq.append([int2token[num_label] for num_label in seq])

    # get list for aa string list
    aa_seq = []
    for seq in temp_aa_seq:
        aa_seq.append(''.join(seq).replace('-',''))

    return aa_seq
